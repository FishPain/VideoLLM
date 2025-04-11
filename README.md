# VideoLLM

> Note: 83 Qns unable to be answered due to video not being available.
> Output proposed to put as "Not Available" in the final submission.
> Alternatively let model make a smart guess.

## Development Environment
> H100 SXM x1, 80GB GPU-RAM, 100GB Disk

## Model Performance (Correctness, Robustness)
- Qwen2.5vl 7b with prompt tuning: 27.47%, 5.5%
- Qwen2.5vl 32b with prompt tuning: 40.33%, 23%
- Qwen2.5vl 7b with prompt tuning + Hallucinate Missing Video: 32.87%, 5.8% (Base Config)
- Qwen2.5vl 7b with Base + Increase fps (1->5): Not tested
- Qwen2.5vl 7b Base + Evaluator: Not Tested
- Qwen2.5vl 7b Base + Audio Captioning / Whisper: Not Tested
- Qwen2.5vl 7b Base + (Adapter using RNN to do recurrsive video evaluation): Not Tested
