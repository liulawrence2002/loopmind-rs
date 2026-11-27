//! `loopmind serve`: a local chat page backed by this repo's own inference engine.
//!
//! One std `TcpListener` serves the page and a small JSON API. A worker thread owns
//! the tokenizer and the loaded checkpoint and answers one chat request at a time.
//! Connection threads parse requests and stream the worker's events back as NDJSON.
//! When the browser goes away (or the user presses Stop), the next write fails, the
//! worker's send fails, and generation stops.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::{Duration, Instant};

use loopmind_model::{GenerateOptions, PretrainEngine};
use loopmind_tokenizer::{ChatTurn, LoopTokenizer, Role, TokenizerError};
use serde::Deserialize;
use serde_json::{json, Value};

const PAGE: &str = include_str!("chat.html");
const MAX_BODY: usize = 1 << 20;
const MAX_LOOPS: usize = 64;

/// A checkpoint the page can pick: a run directory under `runs/`.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub name: String,
    pub path: PathBuf,
    pub step: Option<u64>,
    /// `nll` and `recommended` from `runs/eval/<name>.json`, when that report exists.
    pub eval: Option<Value>,
}

/// What the page starts with.
#[derive(Debug, Clone)]
pub struct Defaults {
    pub checkpoint: String,
    pub loops: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub max_new: usize,
    /// Where the defaults came from, shown on the page.
    pub source: String,
}

pub struct ServeOptions {
    pub host: String,
    pub port: u16,
    pub device: String,
    pub checkpoints: Vec<Checkpoint>,
    pub defaults: Defaults,
}

/// Every directory in `runs` that has a `latest.json`, sorted by name.
pub fn discover(runs: &Path) -> Vec<Checkpoint> {
    let Ok(entries) = std::fs::read_dir(runs) else {
        return Vec::new();
    };
    let mut found: Vec<Checkpoint> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.join("latest.json").is_file())
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?.to_string();
            let step = read_json(&path.join("latest.json"))
                .and_then(|latest| latest.get("step").and_then(Value::as_u64));
            let eval = read_json(&runs.join("eval").join(format!("{name}.json"))).map(|report| {
                json!({ "nll": report.get("nll"), "recommended": report.get("recommended") })
            });
            Some(Checkpoint {
                name,
                path,
                step,
                eval,
            })
        })
        .collect();
    found.sort_by(|left, right| left.name.cmp(&right.name));
    found
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// Render `turns` with a generation prompt. Drops the oldest non-system turns until the
/// prompt leaves `max_new` tokens of the context, keeping at least the last turn.
/// Returns the prompt ids and how many turns were dropped from the front of `turns`.
pub fn fit_prompt(
    tokenizer: &LoopTokenizer,
    turns: &mut Vec<ChatTurn>,
    max_new: usize,
    context: usize,
) -> Result<(Vec<u32>, usize), TokenizerError> {
    let keep = usize::from(turns.first().is_some_and(|turn| turn.role == Role::System));
    let mut dropped = 0;
    let mut prompt = tokenizer.render_chat(turns, true)?;
    while prompt.ids.len() + max_new > context && turns.len() > keep + 1 {
        turns.remove(keep);
        dropped += 1;
        prompt = tokenizer.render_chat(turns, true)?;
    }
    Ok((prompt.ids, dropped))
}

/// Bind, load the starting checkpoint, and serve until the process is stopped.
pub fn serve(
    options: ServeOptions,
    tokenizer: LoopTokenizer,
    stop: [u32; 2],
) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind((options.host.as_str(), options.port))?;
    let address = listener.local_addr()?;
    let info = Arc::new(info_json(&options).to_string());

    let (jobs, inbox) = mpsc::channel::<Job>();
    let (ready, started) = mpsc::channel::<Result<f64, String>>();
    let worker = Worker {
        tokenizer,
        stop,
        checkpoints: options.checkpoints.clone(),
        device: options.device.clone(),
        loaded: None,
    };
    let first = options.defaults.checkpoint.clone();
    thread::spawn(move || worker.run(&first, inbox, ready));
    let seconds = started.recv()??;
    eprintln!(
        "loopmind serve: loaded {} on {} in {seconds:.1} s",
        options.defaults.checkpoint, options.device
    );
    eprintln!("open http://{address}/   (Ctrl-C stops the server)");

    let loopback = address.ip().is_loopback();
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let jobs = jobs.clone();
        let info = Arc::clone(&info);
        let port = address.port();
        thread::spawn(move || {
            if let Err(error) = handle(stream, &jobs, &info, loopback.then_some(port)) {
                if !matches!(
                    error.kind(),
                    io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
                ) {
                    eprintln!("serve: {error}");
                }
            }
        });
    }
    Ok(())
}

fn info_json(options: &ServeOptions) -> Value {
    let defaults = &options.defaults;
    json!({
        "device": options.device,
        "checkpoints": options.checkpoints.iter().map(|checkpoint| json!({
            "name": checkpoint.name,
            "step": checkpoint.step,
            "eval": checkpoint.eval,
        })).collect::<Vec<_>>(),
        "defaults": {
            "checkpoint": defaults.checkpoint,
            "loops": defaults.loops,
            "temperature": defaults.temperature,
            "top_k": defaults.top_k,
            "repetition_penalty": defaults.repetition_penalty,
            "max_new": defaults.max_new,
            "source": defaults.source,
        },
        "max_loops": MAX_LOOPS,
    })
}

/// One chat request from the page.
#[derive(Debug, Deserialize)]
struct ChatRequest {
    checkpoint: String,
    #[serde(default)]
    system: String,
    messages: Vec<Message>,
    loops: usize,
    temperature: f32,
    top_k: usize,
    repetition_penalty: f32,
    max_new: usize,
    #[serde(default)]
    seed: u64,
}

#[derive(Debug, Deserialize)]
struct Message {
    role: String,
    content: String,
}

impl ChatRequest {
    fn turns(&self) -> Result<Vec<ChatTurn>, String> {
        let mut turns = Vec::with_capacity(self.messages.len() + 1);
        if !self.system.trim().is_empty() {
            turns.push(ChatTurn::new(Role::System, self.system.trim()));
        }
        for message in &self.messages {
            let role = match message.role.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                other => return Err(format!("unknown role {other:?}")),
            };
            turns.push(ChatTurn::new(role, message.content.as_str()));
        }
        match turns.last() {
            Some(turn) if turn.role == Role::User => Ok(turns),
            _ => Err("the conversation must end with a user message".into()),
        }
    }

    fn options(&self, context: usize) -> Result<GenerateOptions, String> {
        if !(1..=MAX_LOOPS).contains(&self.loops) {
            return Err(format!("loops must be 1 to {MAX_LOOPS}"));
        }
        if !(1..=context).contains(&self.max_new) {
            return Err(format!("max_new must be 1 to {context}"));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("temperature must be 0 or more".into());
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err("repetition_penalty must be above 0".into());
        }
        Ok(GenerateOptions {
            loops: self.loops,
            max_new: self.max_new,
            temperature: self.temperature,
            top_k: self.top_k,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
        })
    }
}

/// A chat request and where to send its NDJSON events.
struct Job {
    request: ChatRequest,
    events: mpsc::Sender<String>,
}

/// Owns the tokenizer and at most one loaded checkpoint.
struct Worker {
    tokenizer: LoopTokenizer,
    stop: [u32; 2],
    checkpoints: Vec<Checkpoint>,
    device: String,
    loaded: Option<(String, PretrainEngine)>,
}

impl Worker {
    fn run(
        mut self,
        first: &str,
        inbox: mpsc::Receiver<Job>,
        ready: mpsc::Sender<Result<f64, String>>,
    ) {
        let started = Instant::now();
        let loaded = self.load(first).map(|()| started.elapsed().as_secs_f64());
        let failed = loaded.is_err();
        let _ = ready.send(loaded);
        if failed {
            return;
        }
        for job in inbox {
            if let Err(message) = self.answer(&job) {
                let _ = job
                    .events
                    .send(json!({ "type": "error", "message": message }).to_string());
            }
        }
    }

    fn load(&mut self, name: &str) -> Result<(), String> {
        if self
            .loaded
            .as_ref()
            .is_some_and(|(loaded, _)| loaded == name)
        {
            return Ok(());
        }
        let checkpoint = self
            .checkpoints
            .iter()
            .find(|checkpoint| checkpoint.name == name)
            .ok_or_else(|| format!("unknown checkpoint {name:?}"))?;
        // Free the old weights before loading the new ones.
        self.loaded = None;
        let engine = loopmind_train::load_inference(&checkpoint.path, &self.device)
            .map_err(|error| format!("loading {name}: {error}"))?;
        self.loaded = Some((name.to_string(), engine));
        Ok(())
    }

    fn answer(&mut self, job: &Job) -> Result<(), String> {
        let request = &job.request;
        let started = Instant::now();
        self.load(&request.checkpoint)?;
        let load_seconds = started.elapsed().as_secs_f64();
        let Some((name, engine)) = &self.loaded else {
            return Err("no checkpoint is loaded".into());
        };
        let context = engine.config().context_length;
        let options = request.options(context)?;
        let mut turns = request.turns()?;
        let (prompt, dropped) = fit_prompt(&self.tokenizer, &mut turns, options.max_new, context)
            .map_err(|error| error.to_string())?;
        let start = json!({
            "type": "start",
            "checkpoint": name,
            "prompt_tokens": prompt.len(),
            "dropped_turns": dropped,
            "load_seconds": load_seconds,
        });
        if job.events.send(start.to_string()).is_err() {
            return Ok(());
        }

        let begun = Instant::now();
        let mut so_far = Vec::new();
        let mut cancelled = false;
        let tokenizer = &self.tokenizer;
        let ids = engine.generate_stream(prompt, &options, &self.stop, &mut |id| {
            so_far.push(id);
            // Send the whole decode each time: a token can end mid-character, and the
            // next decode fixes it.
            let text = tokenizer.decode(&so_far).unwrap_or_default();
            let event = json!({ "type": "text", "text": text, "tokens": so_far.len() });
            cancelled = job.events.send(event.to_string()).is_err();
            !cancelled
        });
        let seconds = begun.elapsed().as_secs_f64();
        let stop = if cancelled {
            "cancelled"
        } else if ids.len() >= options.max_new {
            "length"
        } else {
            "end"
        };
        let text = tokenizer
            .decode(&ids)
            .map_err(|error| error.to_string())?
            .trim()
            .to_string();
        let done = json!({
            "type": "done",
            "text": text,
            "tokens": ids.len(),
            "seconds": seconds,
            "tokens_per_second": if seconds > 0.0 { ids.len() as f64 / seconds } else { 0.0 },
            "loops": options.loops,
            "stop": stop,
        });
        let _ = job.events.send(done.to_string());
        Ok(())
    }
}

struct HttpRequest {
    method: String,
    path: String,
    host: Option<String>,
    body: Vec<u8>,
}

fn handle(
    mut stream: TcpStream,
    jobs: &mpsc::Sender<Job>,
    info: &str,
    loopback_port: Option<u16>,
) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_nodelay(true)?;
    let Some(request) = read_request(&stream)? else {
        return Ok(());
    };
    // Bound to loopback: refuse requests addressed to any other host name, so a web
    // page cannot reach the server through DNS rebinding.
    if let Some(port) = loopback_port {
        if !request
            .host
            .as_deref()
            .is_some_and(|host| is_local_host(host, port))
        {
            return respond(
                &mut stream,
                "403 Forbidden",
                "text/plain; charset=utf-8",
                b"forbidden host\n",
            );
        }
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/" | "/index.html") => respond(
            &mut stream,
            "200 OK",
            "text/html; charset=utf-8",
            PAGE.as_bytes(),
        ),
        ("GET", "/api/info") => respond(&mut stream, "200 OK", "application/json", info.as_bytes()),
        ("POST", "/api/chat") => chat(stream, jobs, &request.body),
        ("GET", "/favicon.ico") => respond(&mut stream, "204 No Content", "text/plain", b""),
        _ => respond(
            &mut stream,
            "404 Not Found",
            "text/plain; charset=utf-8",
            b"not found\n",
        ),
    }
}

fn is_local_host(host: &str, port: u16) -> bool {
    ["localhost", "127.0.0.1", "[::1]"]
        .iter()
        .any(|name| host == *name || host == format!("{name}:{port}"))
}

fn chat(mut stream: TcpStream, jobs: &mpsc::Sender<Job>, body: &[u8]) -> io::Result<()> {
    let request: ChatRequest = match serde_json::from_slice(body) {
        Ok(request) => request,
        Err(error) => {
            let message = format!("bad chat request: {error}\n");
            return respond(
                &mut stream,
                "400 Bad Request",
                "text/plain; charset=utf-8",
                message.as_bytes(),
            );
        }
    };
    let (events, inbox) = mpsc::channel();
    if jobs.send(Job { request, events }).is_err() {
        return respond(
            &mut stream,
            "503 Service Unavailable",
            "text/plain; charset=utf-8",
            b"the model worker stopped\n",
        );
    }
    stream.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
    )?;
    for event in inbox {
        stream.write_all(event.as_bytes())?;
        stream.write_all(b"\n")?;
        stream.flush()?;
    }
    Ok(())
}

fn read_request(stream: &TcpStream) -> io::Result<Option<HttpRequest>> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or("/");
    let path = target.split('?').next().unwrap_or("/").to_string();
    let mut length = 0usize;
    let mut host = None;
    for _ in 0..100 {
        let mut header = String::new();
        if reader.read_line(&mut header)? == 0 {
            break;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        let Some((name, value)) = header.split_once(':') else {
            continue;
        };
        let (name, value) = (name.trim(), value.trim());
        if name.eq_ignore_ascii_case("content-length") {
            length = value
                .parse()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad Content-Length"))?;
        } else if name.eq_ignore_ascii_case("host") {
            host = Some(value.to_string());
        }
    }
    if length > MAX_BODY {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "request body too large",
        ));
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Some(HttpRequest {
        method,
        path,
        host,
        body,
    }))
}

fn respond(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(messages: &[(&str, &str)]) -> ChatRequest {
        ChatRequest {
            checkpoint: "sft".into(),
            system: String::new(),
            messages: messages
                .iter()
                .map(|(role, content)| Message {
                    role: role.to_string(),
                    content: content.to_string(),
                })
                .collect(),
            loops: 4,
            temperature: 0.7,
            top_k: 40,
            repetition_penalty: 1.1,
            max_new: 64,
            seed: 0,
        }
    }

    #[test]
    fn requests_must_end_on_a_user_turn_and_stay_in_range() {
        assert_eq!(request(&[("user", "hi")]).turns().unwrap().len(), 1);
        assert!(request(&[("user", "hi"), ("assistant", "hello")])
            .turns()
            .is_err());
        assert!(request(&[("robot", "hi")]).turns().is_err());
        assert!(request(&[]).turns().is_err());
        let mut with_system = request(&[("user", "hi")]);
        with_system.system = "  Be brief.  ".into();
        let turns = with_system.turns().unwrap();
        assert_eq!(turns[0], ChatTurn::new(Role::System, "Be brief."));

        let mut bad = request(&[("user", "hi")]);
        bad.loops = 0;
        assert!(bad.options(1024).is_err());
        bad.loops = 4;
        bad.max_new = 2000;
        assert!(bad.options(1024).is_err());
        bad.max_new = 64;
        bad.temperature = -1.0;
        assert!(bad.options(1024).is_err());
        assert!(request(&[("user", "hi")]).options(1024).is_ok());
    }

    #[test]
    fn local_host_names_only() {
        assert!(is_local_host("127.0.0.1:8080", 8080));
        assert!(is_local_host("localhost:8080", 8080));
        assert!(is_local_host("localhost", 8080));
        assert!(!is_local_host("evil.example:8080", 8080));
        assert!(!is_local_host("localhost:9090", 8080));
    }

    #[test]
    fn fit_prompt_drops_the_oldest_turns_but_keeps_the_system_turn() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tokenizer/tokenizer.json");
        let tokenizer = LoopTokenizer::from_file(&path).unwrap();
        let mut turns = vec![
            ChatTurn::new(Role::System, "Be brief."),
            ChatTurn::new(Role::User, "first question ".repeat(20)),
            ChatTurn::new(Role::Assistant, "first answer ".repeat(20)),
            ChatTurn::new(Role::User, "second"),
        ];
        let len = |turns: &[ChatTurn]| tokenizer.render_chat(turns, true).unwrap().ids.len();
        let without_first = len(&[turns[0].clone(), turns[2].clone(), turns[3].clone()]);
        let minimal = len(&[turns[0].clone(), turns[3].clone()]);

        // Room for the answer once the first user turn is gone.
        let mut fits_one = turns.clone();
        let (ids, dropped) = fit_prompt(&tokenizer, &mut fits_one, 16, without_first + 16).unwrap();
        assert_eq!((dropped, ids.len()), (1, without_first));
        assert_eq!(fits_one[1].role, Role::Assistant);

        // Only the system turn and the last user turn fit.
        let (ids, dropped) = fit_prompt(&tokenizer, &mut turns, 16, minimal + 16).unwrap();
        assert_eq!((dropped, ids.len()), (2, minimal));
        assert_eq!(turns[0].role, Role::System);
        assert_eq!(turns.last().unwrap().content, "second");

        // Nothing else can go: the system turn and the last turn always stay.
        let mut tight = turns.clone();
        let (_, dropped) = fit_prompt(&tokenizer, &mut tight, 16, 1).unwrap();
        assert_eq!((dropped, tight.len()), (0, 2));

        let mut short = vec![ChatTurn::new(Role::User, "hi")];
        let (_, dropped) = fit_prompt(&tokenizer, &mut short, 16, 1024).unwrap();
        assert_eq!(dropped, 0);
    }
}
