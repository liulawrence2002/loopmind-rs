//! Answer checks for the four held-out eval configs.
//!
//! The eval split holds out checkable rows: `math_reasoning`, `logic_reasoning`,
//! `science_knowledge`, and `structured_output`. Each check compares the part of an
//! answer that can be compared mechanically. Rows whose reference has no such part
//! are unscored rather than wrong.

/// Outcome of one answer check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Correct,
    Wrong,
    /// The reference has nothing mechanical to compare against.
    Unscored,
}

/// Check `answer` against `reference` for the eval config the row came from.
pub fn score(config: &str, reference: &str, answer: &str) -> Verdict {
    match config {
        "math_reasoning" => score_math(reference, answer),
        "logic_reasoning" => score_logic(reference, answer),
        "science_knowledge" => score_choice(reference, answer),
        "structured_output" => score_json(reference, answer),
        _ => Verdict::Unscored,
    }
}

/// Whether `answer` contains a parseable JSON object or array.
pub fn json_valid(answer: &str) -> bool {
    first_json(answer).is_some()
}

/// Final numbers must agree. The final number follows `Final answer:`, `The answer is:`,
/// or sits in `\boxed{}`; without a marker it is the last number in the text.
fn score_math(reference: &str, answer: &str) -> Verdict {
    let Some(want) = final_number(reference) else {
        return Verdict::Unscored;
    };
    match final_number(answer) {
        Some(got) if numbers_equal(&want, &got) => Verdict::Correct,
        _ => Verdict::Wrong,
    }
}

fn final_number(text: &str) -> Option<String> {
    for marker in ["Final answer:", "The answer is:", "The answer is"] {
        if let Some(index) = text.rfind(marker) {
            if let Some(number) = numbers(&text[index + marker.len()..]).into_iter().next() {
                return Some(number);
            }
        }
    }
    if let Some(index) = text.rfind("\\boxed{") {
        let tail = &text[index + "\\boxed{".len()..];
        let inside = tail.split('}').next().unwrap_or("");
        if let Some(number) = numbers(inside).into_iter().next() {
            return Some(number);
        }
    }
    numbers(text).pop()
}

/// Numbers in `text`, with thousands separators and a trailing period removed.
fn numbers(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut found = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let negative = chars[index] == '-'
            && chars.get(index + 1).is_some_and(char::is_ascii_digit)
            && (index == 0 || !chars[index - 1].is_ascii_alphanumeric());
        if chars[index].is_ascii_digit() || negative {
            let start = index;
            index += 1;
            while index < chars.len()
                && (chars[index].is_ascii_digit()
                    || ((chars[index] == ',' || chars[index] == '.')
                        && chars.get(index + 1).is_some_and(char::is_ascii_digit)))
            {
                index += 1;
            }
            let token: String = chars[start..index]
                .iter()
                .filter(|ch| **ch != ',')
                .collect();
            found.push(token);
        } else {
            index += 1;
        }
    }
    found
}

fn numbers_equal(left: &str, right: &str) -> bool {
    match (left.parse::<f64>(), right.parse::<f64>()) {
        (Ok(left), Ok(right)) => (left - right).abs() <= 1e-6 * left.abs().max(1.0),
        _ => left == right,
    }
}

/// The text after the last `Answer:`, compared case- and whitespace-insensitively.
fn score_logic(reference: &str, answer: &str) -> Verdict {
    let Some(want) = after_last(reference, "Answer:") else {
        return Verdict::Unscored;
    };
    match after_last(answer, "Answer:") {
        Some(got) if normalize(&got) == normalize(&want) => Verdict::Correct,
        _ => Verdict::Wrong,
    }
}

fn after_last(text: &str, marker: &str) -> Option<String> {
    text.rfind(marker)
        .map(|index| text[index + marker.len()..].trim().to_string())
        .filter(|tail| !tail.is_empty())
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_end_matches('.')
        .to_lowercase()
}

/// Multiple choice: a reference like `B. deep ocean` is scored by its letter.
/// The model's choice is the first `X.` that opens the answer or follows whitespace.
fn score_choice(reference: &str, answer: &str) -> Verdict {
    let Some(want) = choice_letter(reference.trim()).filter(|_| reference.trim().len() < 120)
    else {
        return Verdict::Unscored;
    };
    let got = answer.trim();
    let picked = got
        .char_indices()
        .filter(|(index, _)| *index == 0 || got[..*index].ends_with(char::is_whitespace))
        .find_map(|(index, _)| choice_letter(&got[index..]));
    if picked == Some(want) {
        Verdict::Correct
    } else {
        Verdict::Wrong
    }
}

fn choice_letter(text: &str) -> Option<char> {
    let mut chars = text.chars();
    let letter = chars.next()?;
    let dot = chars.next()?;
    (letter.is_ascii_uppercase() && letter <= 'H' && dot == '.').then_some(letter)
}

/// The first JSON value in the answer must equal the reference's.
fn score_json(reference: &str, answer: &str) -> Verdict {
    let Some(want) = first_json(reference) else {
        return Verdict::Unscored;
    };
    match first_json(answer) {
        Some(got) if got == want => Verdict::Correct,
        _ => Verdict::Wrong,
    }
}

/// The first object or array that parses, ignoring code fences and leading prose.
fn first_json(text: &str) -> Option<serde_json::Value> {
    for (index, ch) in text.char_indices() {
        if ch != '{' && ch != '[' {
            continue;
        }
        let mut stream =
            serde_json::Deserializer::from_str(&text[index..]).into_iter::<serde_json::Value>();
        if let Some(Ok(value)) = stream.next() {
            if value.is_object() || value.is_array() {
                return Some(value);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn math_compares_the_final_number() {
        let reference = "So it will take him 84/7=12 days.\n\nFinal answer: 12";
        assert_eq!(
            score("math_reasoning", reference, "12 days.\nFinal answer: 12"),
            Verdict::Correct
        );
        assert_eq!(
            score("math_reasoning", reference, "Final answer: 1,200"),
            Verdict::Wrong
        );
        assert_eq!(
            score(
                "math_reasoning",
                "the ball lands after $\\boxed{2.5}$ seconds.\nThe answer is: 2.5",
                "it takes 2.50 seconds"
            ),
            Verdict::Correct
        );
        assert_eq!(
            score(
                "math_reasoning",
                "So, the first laptop costs $500.",
                "It costs $2,000 then $500."
            ),
            Verdict::Correct
        );
        assert_eq!(
            score("math_reasoning", "no digits", "12"),
            Verdict::Unscored
        );
    }

    #[test]
    fn logic_compares_the_answer_line() {
        let reference = "Steps...\n\nAnswer: Mateo → water, Priya → coffee, Diego → tea";
        assert_eq!(
            score(
                "logic_reasoning",
                reference,
                "Answer:  mateo → water,  Priya → coffee, Diego → tea."
            ),
            Verdict::Correct
        );
        assert_eq!(
            score("logic_reasoning", reference, "Mateo → water"),
            Verdict::Wrong
        );
        assert_eq!(
            score("logic_reasoning", "- `s[0:12]` → small steps", "x"),
            Verdict::Unscored
        );
    }

    #[test]
    fn science_scores_multiple_choice_by_letter() {
        assert_eq!(
            score("science_knowledge", "B. deep ocean", "B. deep ocean"),
            Verdict::Correct
        );
        assert_eq!(
            score(
                "science_knowledge",
                "B. deep ocean",
                "The answer is B. deep ocean"
            ),
            Verdict::Correct
        );
        assert_eq!(
            score("science_knowledge", "B. deep ocean", "C. desert"),
            Verdict::Wrong
        );
        assert_eq!(
            score(
                "science_knowledge",
                "Animalcule is an old term for microscopic organisms.",
                "B."
            ),
            Verdict::Unscored
        );
    }

    #[test]
    fn json_must_parse_and_match() {
        let reference = "{\n  \"name\": \"Grace\",\n  \"salary\": 133000\n}";
        assert_eq!(
            score(
                "structured_output",
                reference,
                "```json\n{\"salary\": 133000, \"name\": \"Grace\"}\n```"
            ),
            Verdict::Correct
        );
        assert_eq!(
            score("structured_output", reference, "{\"name\": \"Grace\"}"),
            Verdict::Wrong
        );
        assert!(json_valid("Here: [1, 2]"));
        assert!(!json_valid("{\"name\": "));
    }
}
