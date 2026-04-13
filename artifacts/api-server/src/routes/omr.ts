import { Router, type IRouter, type Request, type Response } from "express";
import axios from "axios";
import { execFile, type ExecFileException } from "child_process";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import { logger } from "../lib/logger";

const router: IRouter = Router();

const NEET_DIR = path.resolve(__dirname, "../../../neet-omr-checker");
const SCRIPT = path.join(NEET_DIR, "omr_api_cli.py");
const PYTHON_BIN = process.env.PYTHON_BIN ?? "python";

async function downloadImage(url: string): Promise<string> {
  const tmpPath = path.join(NEET_DIR, `tmp_${crypto.randomUUID()}.jpg`);
  const response = await axios.get(url, { responseType: "arraybuffer" });
  fs.writeFileSync(tmpPath, response.data);
  return tmpPath;
}

function runPython(imagePath: string, useAi: boolean): Promise<object[]> {
  return new Promise((resolve, reject) => {
    execFile(
      PYTHON_BIN,
      [SCRIPT, "--image", imagePath, ...(useAi ? ["--use-ai"] : [])],
      { maxBuffer: 1024 * 1024 * 10, cwd: NEET_DIR },
      (err: ExecFileException | null, stdout: string, stderr: string) => {
        if (err != null) {
          logger.error({ stderr }, "Python script error");
          return reject(new Error(stderr || err.message));
        }
        try {
          resolve(JSON.parse(stdout.trim()));
        } catch {
          logger.error({ stdout }, "Failed to parse Python output");
          reject(new Error("Invalid JSON from Python"));
        }
      },
    );
  });
}

router.post("/omr/predict", async (req: Request, res: Response) => {
  let tmpPath: string | null = null;
  try {
    const { imageUrl, useAi = false } = req.body as { imageUrl?: unknown; useAi?: unknown };

    if (!imageUrl || typeof imageUrl !== "string") {
      return res.status(400).json({ success: false, error: "imageUrl is required" });
    }

    if (typeof useAi !== "boolean") {
      return res.status(400).json({ success: false, error: "useAi must be a boolean" });
    }

    logger.info({ imageUrl, useAi }, "OMR predict request received");

    tmpPath = await downloadImage(imageUrl);

    const answers = await runPython(tmpPath, useAi);
    return res.json({ success: true, answers });
  } catch (err: any) {
    logger.error({ err }, "OMR predict failed");
    return res.status(500).json({ success: false, error: err?.message ?? "Unknown error" });
  } finally {
    if (tmpPath && fs.existsSync(tmpPath)) {
      fs.unlinkSync(tmpPath);
    }
  }
});

export default router;
