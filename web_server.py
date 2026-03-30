"""OCR-Excel 웹 서버 — Railway 배포용.

파일 업로드 → OCR 변환 → Excel 다운로드
"""

import os
import tempfile
import uuid

from flask import Flask, request, send_file, jsonify, render_template_string

from core.pipeline import run_pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB

UPLOAD_DIR = tempfile.mkdtemp()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR-Excel 변환기</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, 'Malgun Gothic', sans-serif; background: #f5f7fa; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .container { background: white; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); padding: 48px; max-width: 520px; width: 90%; }
        h1 { font-size: 24px; color: #1a1a2e; margin-bottom: 8px; }
        .sub { color: #666; font-size: 14px; margin-bottom: 32px; }
        .drop-zone { border: 2px dashed #ccc; border-radius: 12px; padding: 48px 24px; text-align: center; cursor: pointer; transition: all 0.2s; }
        .drop-zone:hover, .drop-zone.drag-over { border-color: #002060; background: #f0f4ff; }
        .drop-zone p { color: #888; font-size: 15px; }
        .drop-zone .icon { font-size: 48px; margin-bottom: 12px; }
        input[type=file] { display: none; }
        .btn { display: inline-block; background: #002060; color: white; padding: 12px 32px; border-radius: 8px; font-size: 15px; border: none; cursor: pointer; margin-top: 16px; text-decoration: none; }
        .btn:hover { background: #003090; }
        .btn:disabled { background: #999; cursor: not-allowed; }
        .status { margin-top: 20px; padding: 12px; border-radius: 8px; font-size: 14px; display: none; }
        .status.loading { display: block; background: #fff3cd; color: #856404; }
        .status.success { display: block; background: #d4edda; color: #155724; }
        .status.error { display: block; background: #f8d7da; color: #721c24; }
        .info { margin-top: 24px; font-size: 12px; color: #999; }
        .engines { display: flex; gap: 8px; margin-top: 16px; flex-wrap: wrap; }
        .engines label { font-size: 13px; color: #555; display: flex; align-items: center; gap: 4px; }
    </style>
</head>
<body>
<div class="container">
    <h1>OCR-Excel 변환기</h1>
    <p class="sub">채권 수요예측표 / 재무제표 이미지를 Excel로 변환합니다</p>

    <form id="uploadForm">
        <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <div class="icon">📄</div>
            <p>파일을 여기에 드래그하거나 클릭하여 선택</p>
            <p style="font-size:12px; color:#aaa; margin-top:8px">PNG, JPG, PDF (최대 20MB)</p>
        </div>
        <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.pdf,.bmp,.tiff,.tif,.webp">
        <div class="engines">
            <label><input type="checkbox" id="useAi" checked> AI Vision 사용</label>
        </div>
        <button type="submit" class="btn" id="submitBtn" disabled>변환하기</button>
    </form>

    <div class="status" id="status"></div>
    <div id="downloadLink"></div>

    <div class="info">
        지원 엔진: PaddleOCR (로컬) → Groq (무료) → GPT-4o-mini (유료) → Gemini
    </div>
</div>

<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const submitBtn = document.getElementById('submitBtn');
const status = document.getElementById('status');
const downloadLink = document.getElementById('downloadLink');

['dragover','dragenter'].forEach(e => {
    dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.add('drag-over'); });
});
['dragleave','drop'].forEach(e => {
    dropZone.addEventListener(e, ev => { ev.preventDefault(); dropZone.classList.remove('drag-over'); });
});
dropZone.addEventListener('drop', ev => {
    fileInput.files = ev.dataTransfer.files;
    fileInput.dispatchEvent(new Event('change'));
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        dropZone.querySelector('p').textContent = fileInput.files[0].name;
        submitBtn.disabled = false;
    }
});

document.getElementById('uploadForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const file = fileInput.files[0];
    if (!file) return;

    submitBtn.disabled = true;
    status.className = 'status loading';
    status.textContent = '변환 중... (30초~2분 소요)';
    status.style.display = 'block';
    downloadLink.innerHTML = '';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('use_ai', document.getElementById('useAi').checked ? '1' : '0');

    try {
        const resp = await fetch('/api/convert', { method: 'POST', body: formData });
        if (resp.ok) {
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const fname = resp.headers.get('X-Filename') || 'result.xlsx';
            status.className = 'status success';
            status.textContent = '변환 완료!';
            downloadLink.innerHTML = '<a class="btn" href="' + url + '" download="' + fname + '">다운로드 (' + fname + ')</a>';
        } else {
            const err = await resp.json();
            status.className = 'status error';
            status.textContent = '오류: ' + (err.error || '알 수 없는 오류');
        }
    } catch (e) {
        status.className = 'status error';
        status.textContent = '네트워크 오류: ' + e.message;
    }
    submitBtn.disabled = false;
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/convert", methods=["POST"])
def convert():
    if "file" not in request.files:
        return jsonify({"error": "파일이 없습니다"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "파일명이 없습니다"}), 400

    use_ai = request.form.get("use_ai", "1") == "1"

    # 임시 파일 저장
    ext = os.path.splitext(file.filename)[1].lower()
    uid = uuid.uuid4().hex[:8]
    input_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
    output_path = os.path.join(UPLOAD_DIR, f"{uid}_result.xlsx")

    file.save(input_path)

    try:
        result_path = run_pipeline(
            input_path,
            output_path=output_path,
            use_ai=use_ai,
        )

        out_name = os.path.splitext(file.filename)[0] + "_변환결과.xlsx"
        response = send_file(result_path, as_attachment=True, download_name=out_name)
        response.headers["X-Filename"] = out_name
        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # 정리
        try:
            os.unlink(input_path)
        except OSError:
            pass


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
