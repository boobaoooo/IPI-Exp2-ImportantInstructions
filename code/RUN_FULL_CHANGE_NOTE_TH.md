# run_full.py — Change Note for P1 (fingerprint 5dda6437 → 36e2dae4)

Fingerprint ทั้งหมดเป็น SHA256 แบบ newline-normalised (CRLF→LF) ค่าตรงกันทั้ง Windows/Linux
- เวอร์ชันที่ P1 เคยตรวจ: `5dda643709a86bf1e5d7ac130f891b794756079b786f63df431a185afc8a84c9`
- เวอร์ชันปัจจุบัน (ที่ใช้รัน 3-run และจะใช้รัน batch): `36e2dae422520334eae243b6b5c812b6a44af93a81e01f24d51b2ce8077cc3e6`

ไฟล์แนบ: run_full.py (ปัจจุบัน) + run_full_DIFF_5dda6437_to_36e2dae4.patch (unified diff)

## สรุปสั้น
การเปลี่ยนแปลงทั้งหมดอยู่ที่ **ชั้น driver/หลักฐาน (execution-layer)** ไม่แตะ **ตรรกะเชิงวิทยาศาสตร์** ของการรันแต่ละ run
runner_version ยังเป็น 0.2.2 เท่าเดิม และตัว runner (p2runner/*) ไม่เปลี่ยน

## หลักฐานว่าการรันแต่ละ run เหมือนเดิมทุกบิต
1. การเรียกจริงที่ประมวลผลวิทยาศาสตร์คือ `execute_run(...)` ซึ่ง **import มาจาก `p2runner.runner`** (ไม่ได้นิยามใน run_full.py) — ทั้งสองเวอร์ชัน import เหมือนกัน และ p2runner ไม่เปลี่ยน:
   - p2runner/runner.py  38991e2687a5805dd...  (ไม่เปลี่ยน)
   - p2runner/adapters.py 2e98b1276832c0cb5...  (ไม่เปลี่ยน)
   - p2runner/scenarios.py d5b7ff8900dc5354a...  (ไม่เปลี่ยน)
   - p2runner/__init__.py 83c19fe9c044c5a33...  (ไม่เปลี่ยน)  [RUNNER_VERSION = 0.2.2]
2. อาร์กิวเมนต์ที่ส่งเข้า execute_run เหมือนกันทุกตัว:
   `execute_run(case, adapter, row["source_llm_id"], cfg, args.out, repetition_id=rep, attempt_id=attempt, run_id=run_id)`
3. ค่า config การรันเหมือนเดิม: `RunConfig(max_steps=15, timeout_s=180)`
4. การสร้างเคส `build_case(...)` **ไม่เปลี่ยน** (ฟังก์ชันเหมือนเดิมทุกบิต) — payload lookup, vectors, env, hash เดิม
5. POST-RUN guard (identity/context/GPU residency) เหมือนเดิมทุกบิต

เปรียบเทียบระดับฟังก์ชัน (25 ฟังก์ชันที่มีร่วมกัน): ไม่มีฟังก์ชันใดถูกลบ, ฟังก์ชันที่ "เปลี่ยน" มีเพียง 3 ตัว = `runtime_gate`, `write_batch_metadata`, `main` (ดูด้านล่าง) ที่เหลือ 22 ตัวเหมือนเดิมทุกบิต รวมถึง build_case

## สิ่งที่เพิ่มเข้ามา (ทั้งหมดเป็น driver/หลักฐาน ไม่ใช่วิทยาศาสตร์)
- `runtime_gate`: เพิ่มด่านตรวจ **อ่านอย่างเดียว** ว่า env `OLLAMA_CONTEXT_LENGTH == 16384` ในหน้าต่างที่รัน (ค่านี้ถูกบันทึกลงทุก record) — เป็นการ "ตรวจ" ไม่ใช่ "ตั้งค่า" ไม่กระทบคำตอบของโมเดล
- `validate_args`: ตรวจความถูกต้องของอาร์กิวเมนต์ก่อนเริ่ม
- `OutputLock`: lock file กัน batch สองตัวเขียนทับโฟลเดอร์เดียวกัน
- `snapshot_attempts` / `ledger_entry` / attempt_ledger.jsonl: บันทึก attempt ทุกครั้งและระบุ attempt ที่ accept (ตาม Data Contract)
- `free_gb` + ด่านตรวจพื้นที่ดิสก์ก่อนเริ่ม
- `main` แยกเป็น `main → _main → _run_batch`: รองรับ resume/chunk/checkpoint/exit-codes โดยลำดับการรันและการเรียก execute_run ยังเหมือนเดิม
- `write_batch_metadata`: เพิ่มฟิลด์ metadata ของ batch (เช่น argument ที่ใช้) ไม่ใช่ข้อมูลผลลัพธ์

## สิ่งที่ "ไม่" เปลี่ยน
model identities, digests, payloads, vectors, splits, prompts, tools, temperature 0.0, seed 42, parallel_tool_calls false, context 16384, max_steps 15, timeout 180s, การสร้าง env ใหม่ต่อ run, และตรรกะการประเมิน (อยู่ใน p2runner ทั้งหมด ไม่เปลี่ยน)

## ผลต่อ 3 runs ที่ accept ไปแล้ว
3 runs ถูกรันด้วยเวอร์ชัน 36e2dae4 นี้ (เวอร์ชันปัจจุบัน) อยู่แล้ว การรัน batch ต่อด้วยเวอร์ชันเดียวกันจึงสอดคล้องกัน ไม่มีการเปลี่ยนโค้ดระหว่าง 3-run กับ batch ที่จะตามมา
