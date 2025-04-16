# 🧠 VideoLLM – AISG Challenge Submission

> ⚠️ **Note:** 83 questions could not be answered due to missing videos.  
> ✅ The model is now instructed to make **intelligent guesses** for these cases to improve overall completeness.

---

## ⚙️ Development Environment

| Resource        | Specification         |
|----------------|------------------------|
| GPU            | NVIDIA H100 SXM (1x)   |
| GPU Memory     | 80 GB                  |
| Disk Space     | 100 GB                 |

---

## 📊 Model Evaluation – Correctness & Robustness

Due to compute limitations, initial evaluations were conducted using **Qwen2.5-VL 7B**.  
Trends suggest the **Qwen2.5-VL 32B** model offers a **~10–15% improvement** in both correctness and robustness.

| Configuration                                                          | Correctness (%) | Robustness (%) |
|------------------------------------------------------------------------|-------|------|
| Qwen2.5-VL 7B                                                          | None  | None |
| Qwen2.5-VL 7B + Prompt Tuning + 1 FPS                                  | 27.47 | 5.5  |
| Qwen2.5-VL 32B + Prompt Tuning + 1 FPS                                 | 40.33 | 23.0 |
| Qwen2.5-VL 7B + Prompt Tuning + 1 FPS + Mock Missing Videos (Base)     | 32.87 | 5.8  |
| Qwen2.5-VL 7B + Base + 5 FPS                                           | 31.67 | 7.2  |
| Qwen2.5-VL 7B + Base + LLM as Judge to Evaluate Top K Answers          | 31.93 | 8.5  |
| Qwen2.5-VL 7B + Base + Audio Captioning via Whisper                    | None | None  |
| Qwen2.5-VL 7B + Base + RNN Adapter for Recursive Video Understanding   | None | None  |

---

## Ideas
- test out video llama 3
- Video at 0.5 time speed
- Multi-step inference with smaller model and lower resolution video to identify video chunk that's relevant then use a larger model to analyze that higher resolution chunk.