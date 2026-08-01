// TEMPORARY review scaffolding for unslothai/unsloth#7709 -- NOT for commit.
//
// Drives the REAL Studio production functions (commands::check_health and the
// startup port validator reached through it) against a fake backend on
// 127.0.0.1 while a poison HTTP proxy is configured in the environment.
//
// Pre-PR behaviour (loopback_http::client without .no_proxy()) must FAIL.
// Post-PR behaviour must SUCCEED.

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::process::Command;

    const CHILD_ENV: &str = "UNSLOTH_PROXY_E2E_CHILD";
    const HEALTH_JSON: &str = r#"{"status":"healthy","service":"Unsloth UI Backend"}"#;

    fn serve(listener: TcpListener, body: &'static str) {
        std::thread::spawn(move || loop {
            let Ok((mut stream, _)) = listener.accept() else {
                continue;
            };
            std::thread::spawn(move || {
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(2)));
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                    .as_bytes(),
                );
                let _ = stream.flush();
            });
        });
    }

    // Scenario 1: a poison proxy that answers 200 with non-backend JSON. If the
    // request is proxied, check_health sees the wrong service and returns false.
    #[test]
    fn check_health_survives_env_proxy_answering_proxy() {
        run_child_or_body("answering", |port| {
            let healthy = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(crate::commands::check_health(port))
                .unwrap();
            assert!(
                healthy,
                "check_health returned false: the loopback request did not reach the backend"
            );
        });
    }

    // Scenario 2: a proxy pointing at a dead port -- the shape of the real bug
    // report (proxy configured, unusable for loopback).
    #[test]
    fn check_health_survives_env_proxy_dead() {
        run_child_or_body("dead", |port| {
            let healthy = tokio::runtime::Runtime::new()
                .unwrap()
                .block_on(crate::commands::check_health(port))
                .unwrap();
            assert!(
                healthy,
                "check_health returned false: the loopback request did not reach the backend"
            );
        });
    }

    fn run_child_or_body(kind: &str, body: impl FnOnce(u16)) {
        if std::env::var_os(CHILD_ENV).is_some() {
            let port: u16 = std::env::var("UNSLOTH_PROXY_E2E_PORT")
                .unwrap()
                .parse()
                .unwrap();
            body(port);
            return;
        }

        let backend = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let backend_port = backend.local_addr().unwrap().port();
        serve(backend, HEALTH_JSON);

        let proxy_url = if kind == "dead" {
            "http://127.0.0.1:1".to_string()
        } else {
            let proxy = TcpListener::bind(("127.0.0.1", 0)).unwrap();
            let proxy_port = proxy.local_addr().unwrap().port();
            serve(proxy, r#"{"status":"proxy","service":"corporate proxy"}"#);
            format!("http://127.0.0.1:{proxy_port}")
        };

        let current_thread = std::thread::current();
        let test_name = current_thread.name().expect("named test thread");
        assert_ne!(test_name, "main", "run without --test-threads=1");

        let status = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test_name, "--nocapture"])
            .env(CHILD_ENV, "1")
            .env("UNSLOTH_PROXY_E2E_PORT", backend_port.to_string())
            .env("HTTP_PROXY", &proxy_url)
            .env("http_proxy", &proxy_url)
            .env("ALL_PROXY", &proxy_url)
            .env_remove("HTTPS_PROXY")
            .env_remove("https_proxy")
            .env_remove("all_proxy")
            .env_remove("NO_PROXY")
            .env_remove("no_proxy")
            .env_remove("REQUEST_METHOD")
            .status()
            .unwrap();
        assert!(status.success(), "child test failed under {kind} proxy");
    }
}
