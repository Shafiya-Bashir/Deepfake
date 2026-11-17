const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const DEBUG_PORT = 9222;

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function getDebuggerUrl() {
  for (let i = 0; i < 30; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json`);
      const list = await res.json();
      if (list && list.length > 0 && list[0].webSocketDebuggerUrl) {
        return list[0].webSocketDebuggerUrl;
      }
    } catch (e) {
      // Waiting for Chrome CDP
    }
    await sleep(200);
  }
  throw new Error('Could not connect to Chrome debugging port.');
}

async function capture(url, outputPath, options = {}) {
  console.log(`[*] Launching Chrome headless for ${path.basename(outputPath)}...`);
  const chromeProc = spawn(CHROME_PATH, [
    '--headless=new',
    `--remote-debugging-port=${DEBUG_PORT}`,
    `--window-size=${options.width || 1920},${options.height || 1200}`,
    '--hide-scrollbars',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    'about:blank'
  ]);

  try {
    const wsUrl = await getDebuggerUrl();
    console.log(`[*] Connected to Chrome CDP`);

    const ws = new WebSocket(wsUrl);

    let id = 1;
    const callbacks = new Map();

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id && callbacks.has(msg.id)) {
        callbacks.get(msg.id)(msg);
        callbacks.delete(msg.id);
      }
    };

    await new Promise((resolve, reject) => {
      ws.onopen = resolve;
      ws.onerror = reject;
    });

    function send(method, params = {}) {
      return new Promise((resolve) => {
        const msgId = id++;
        callbacks.set(msgId, (res) => resolve(res.result));
        ws.send(JSON.stringify({ id: msgId, method, params }));
      });
    }

    await send('Page.enable');
    await send('Runtime.enable');
    await send('Emulation.setDeviceMetricsOverride', {
      width: options.width || 1920,
      height: options.height || 1200,
      deviceScaleFactor: options.scale || 2,
      mobile: false
    });

    console.log(`[*] Navigating to: ${url}`);
    await send('Page.navigate', { url });

    // Wait for analysis API call and DOM animations to complete
    await sleep(4000);

    console.log(`[*] Capturing screenshot...`);
    const screenshot = await send('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: options.fullPage || false
    });

    if (!screenshot || !screenshot.data) {
      throw new Error('Screenshot data empty');
    }

    const buffer = Buffer.from(screenshot.data, 'base64');
    fs.writeFileSync(outputPath, buffer);
    console.log(`[✓] Screenshot saved: ${outputPath} (${(buffer.length / 1024).toFixed(1)} KB)`);

    ws.close();
  } finally {
    chromeProc.kill('SIGTERM');
  }
}

async function main() {
  const targetDir = path.resolve(__dirname, '../assets/screenshots');
  if (!fs.existsSync(targetDir)) {
    fs.mkdirSync(targetDir, { recursive: true });
  }

  // 1. Capture Dashboard with Critical Fake Detected UI Output
  const fakeOutputPath = path.join(targetDir, 'deepfake_ai_dashboard_fake_detected.png');
  await capture('http://127.0.0.1:8080/?preset=fake_video', fakeOutputPath, {
    width: 1920,
    height: 1200,
    scale: 2
  });

  await sleep(1500);

  // 2. Capture Dashboard with Verified Authentic UI Output
  const realOutputPath = path.join(targetDir, 'deepfake_ai_dashboard_authentic_verified.png');
  await capture('http://127.0.0.1:8080/?preset=real_video', realOutputPath, {
    width: 1920,
    height: 1200,
    scale: 2
  });

  await sleep(1500);

  // 3. Capture Full Page Dashboard View (Complete Telemetry + Live Terminal)
  const fullOutputPath = path.join(targetDir, 'deepfake_ai_dashboard_full_view.png');
  await capture('http://127.0.0.1:8080/?preset=fake_video', fullOutputPath, {
    width: 1920,
    height: 1200,
    scale: 1.5,
    fullPage: true
  });
}

main().catch(err => {
  console.error('[!] Error:', err);
  process.exit(1);
});
