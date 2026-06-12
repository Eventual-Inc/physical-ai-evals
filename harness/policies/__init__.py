"""Policy backends: OpenVLA (baseline) and VLA-JEPA (headliner).

Both are stubs that lazy-import their (mutually incompatible) stacks inside __init__/
act. Import the class only at use-time — and run the two policies in SEPARATE
environments (OpenVLA pins transformers==4.40.1; VLA-JEPA wants the modern LeRobot /
Qwen3-VL stack). See NOTES.md.
"""
