// Behaviour matrix for unslothai/unsloth#7709.
//
// Proves, against the exact reqwest version the Studio crate pins (0.12.28),
// what a DEFAULT client (pre-PR code) and a .no_proxy() client (post-PR code)
// each do when proxy environment variables are present.
//
// Parent process:
//   - binds a target HTTP server on 127.0.0.1 (and ::1 when available) that
//     answers every request with the body TARGET-OK
//   - binds a poison HTTP proxy on 127.0.0.1 that answers every request with
//     the body PROXY-HIT (so "went through the proxy" is distinguishable from
//     "connection refused")
//   - re-executes itself once per (env scenario x target x client) cell, since
//     reqwest samples the proxy environment at Client build time
//
// Child process: performs one request and prints OUTCOME=<kind>|<detail>.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

const CHILD_ENV: &str = "LOOPBACK_MATRIX_CHILD";

fn serve(listener: TcpListener, body: &'static str) {
    std::thread::spawn(move || loop {
        let Ok((mut stream, _)) = listener.accept() else {
            continue;
        };
        std::thread::spawn(move || {
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .as_bytes(),
            );
            let _ = stream.flush();
        });
    });
}

fn child(mode: &str, url: &str) {
    let builder = reqwest::Client::builder().timeout(Duration::from_secs(3));
    let builder = if mode == "noproxy" {
        builder.no_proxy()
    } else {
        builder
    };
    let client = match builder.build() {
        Ok(c) => c,
        Err(e) => {
            println!("OUTCOME=BUILD_ERR|{e}");
            return;
        }
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let out = rt.block_on(async {
        match client.get(url).send().await {
            Ok(r) => {
                let status = r.status().as_u16();
                match r.text().await {
                    Ok(t) => format!("OK|{status} {}", t.trim()),
                    Err(e) => format!("BODY_ERR|{e}"),
                }
            }
            Err(e) => {
                let kind = if e.is_timeout() { "TIMEOUT" } else { "ERR" };
                format!("{kind}|{e}")
            }
        }
    });
    println!("OUTCOME={out}");
}

/// (label, env pairs to set, env names to remove)
fn scenarios(proxy_port: u16) -> Vec<(&'static str, Vec<(String, String)>, Vec<&'static str>)> {
    let p = format!("http://127.0.0.1:{proxy_port}");
    vec![
        ("no proxy env at all", vec![], vec![]),
        (
            "HTTP_PROXY (upper)",
            vec![("HTTP_PROXY".into(), p.clone())],
            vec![],
        ),
        (
            "http_proxy (lower)",
            vec![("http_proxy".into(), p.clone())],
            vec![],
        ),
        (
            "ALL_PROXY",
            vec![("ALL_PROXY".into(), p.clone())],
            vec![],
        ),
        (
            "HTTPS_PROXY only",
            vec![("HTTPS_PROXY".into(), p.clone())],
            vec![],
        ),
        (
            "HTTP_PROXY + NO_PROXY=127.0.0.1",
            vec![
                ("HTTP_PROXY".into(), p.clone()),
                ("NO_PROXY".into(), "127.0.0.1".into()),
            ],
            vec![],
        ),
        (
            "HTTP_PROXY + NO_PROXY=localhost",
            vec![
                ("HTTP_PROXY".into(), p.clone()),
                ("NO_PROXY".into(), "localhost".into()),
            ],
            vec![],
        ),
        (
            "HTTP_PROXY + NO_PROXY=*",
            vec![
                ("HTTP_PROXY".into(), p.clone()),
                ("NO_PROXY".into(), "*".into()),
            ],
            vec![],
        ),
        (
            "HTTP_PROXY + REQUEST_METHOD (CGI)",
            vec![
                ("HTTP_PROXY".into(), p.clone()),
                ("REQUEST_METHOD".into(), "GET".into()),
            ],
            vec![],
        ),
        (
            "HTTP_PROXY=dead port 1 (PR test env)",
            vec![("HTTP_PROXY".into(), "http://127.0.0.1:1".into())],
            vec![],
        ),
        (
            "ALL_PROXY=socks5 dead",
            vec![("ALL_PROXY".into(), "socks5://127.0.0.1:1".into())],
            vec![],
        ),
    ]
}

fn main() {
    if let Ok(_v) = std::env::var(CHILD_ENV) {
        let args: Vec<String> = std::env::args().collect();
        child(&args[1], &args[2]);
        return;
    }

    let target_v4 = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let v4_port = target_v4.local_addr().unwrap().port();
    serve(target_v4, "TARGET-OK");

    let v6_port = match TcpListener::bind(("::1", 0)) {
        Ok(l) => {
            let p = l.local_addr().unwrap().port();
            serve(l, "TARGET-OK");
            Some(p)
        }
        Err(_) => None,
    };

    // A fixed port lets a caller point the OPERATING SYSTEM proxy setting
    // (Windows HKCU Internet Settings, macOS SystemConfiguration) at this
    // listener before launching us, which is how the system-proxy cells below
    // are exercised on those platforms.
    let fixed: u16 = std::env::var("MATRIX_FIXED_PROXY_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let proxy = TcpListener::bind(("127.0.0.1", fixed)).unwrap();
    let proxy_port = proxy.local_addr().unwrap().port();
    serve(proxy, "PROXY-HIT");

    // System-proxy mode: run ONLY the "no proxy environment variable at all"
    // scenario, so whatever routing happens comes from the OS configuration.
    let sysproxy_only = std::env::var_os("MATRIX_SYSPROXY_ONLY").is_some();

    // sanity: the target really is reachable directly
    assert!(TcpStream::connect(("127.0.0.1", v4_port)).is_ok());

    let mut targets: Vec<(String, String)> = vec![
        ("127.0.0.1".into(), format!("http://127.0.0.1:{v4_port}/api/health")),
        ("localhost".into(), format!("http://localhost:{v4_port}/api/health")),
    ];
    if let Some(p6) = v6_port {
        targets.push(("[::1]".into(), format!("http://[::1]:{p6}/api/health")));
    }

    println!("target v4 port = {v4_port}, v6 port = {v6_port:?}, poison proxy port = {proxy_port}");
    println!();
    println!("| # | proxy env | target | default client (pre-PR) | .no_proxy() client (post-PR) |");
    println!("|---|-----------|--------|-------------------------|------------------------------|");

    let mut n = 0;
    let mut regressions = 0;
    let mut sysproxy_default_was_proxied = false;
    let all = scenarios(proxy_port);
    let selected: Vec<_> = if sysproxy_only {
        all.into_iter().take(1).collect()
    } else {
        all
    };
    for (label, sets, removes) in selected {
        for (tname, url) in &targets {
            n += 1;
            let mut results = Vec::new();
            for mode in ["default", "noproxy"] {
                let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
                cmd.args([mode, url]).env(CHILD_ENV, "1");
                for k in [
                    "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY",
                    "all_proxy", "NO_PROXY", "no_proxy", "REQUEST_METHOD",
                ] {
                    cmd.env_remove(k);
                }
                for r in &removes {
                    cmd.env_remove(r);
                }
                for (k, v) in &sets {
                    cmd.env(k, v);
                }
                let out = cmd.output().unwrap();
                let text = String::from_utf8_lossy(&out.stdout);
                let line = text
                    .lines()
                    .find(|l| l.starts_with("OUTCOME="))
                    .unwrap_or("OUTCOME=NO_OUTPUT|")
                    .trim_start_matches("OUTCOME=")
                    .to_string();
                results.push(summarise(&line));
            }
            let post_ok = results[1].starts_with("DIRECT");
            if !post_ok {
                regressions += 1;
            }
            if results[0].starts_with("PROXIED") || results[0].starts_with("FAIL") {
                sysproxy_default_was_proxied = true;
            }
            let label = if sysproxy_only {
                "OS/system proxy only (no env vars)"
            } else {
                label
            };
            println!(
                "| {n} | {label} | {tname} | {} | {} |",
                results[0], results[1]
            );
        }
    }
    println!();
    println!("cells = {n}, post-PR cells NOT reaching the target directly = {regressions}");
    if sysproxy_only {
        println!("sysproxy: default client diverted by the OS setting = {sysproxy_default_was_proxied}");
        // The point of this mode: the OS proxy must actually capture the
        // default client, otherwise the cell proves nothing.
        if !sysproxy_default_was_proxied {
            println!("SYSPROXY_INCONCLUSIVE: the OS proxy setting did not capture the default client");
        } else if regressions == 0 {
            println!("SYSPROXY_CONFIRMED: OS proxy diverts the default client; .no_proxy() still goes direct");
        }
    }
    if regressions > 0 {
        std::process::exit(1);
    }
}

fn summarise(raw: &str) -> String {
    if raw.contains("TARGET-OK") {
        "DIRECT (TARGET-OK)".into()
    } else if raw.contains("PROXY-HIT") {
        "PROXIED (PROXY-HIT)".into()
    } else if raw.starts_with("TIMEOUT") {
        "FAIL (timeout)".into()
    } else if raw.starts_with("ERR") {
        let detail = if raw.contains("Connection refused") || raw.contains("connect") {
            "connect error"
        } else if raw.contains("socks") || raw.contains("SOCKS") {
            "socks error"
        } else {
            "error"
        };
        format!("FAIL ({detail})")
    } else if raw.starts_with("BUILD_ERR") {
        "FAIL (client build)".into()
    } else {
        format!("? ({raw})")
    }
}
