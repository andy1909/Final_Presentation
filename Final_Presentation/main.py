from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles # THÊM IMPORT NÀY
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import json
from fastapi import File, UploadFile, HTTPException, BackgroundTasks
from pathlib import Path
from VRP_0_Main import calculate_distance_matrix, train_vrp_model, run_vrp_inference

app = FastAPI()

# Cấu hình CORS (giữ nguyên hoặc điều chỉnh cho phù hợp)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_UPLOADS_DIR = Path("Uploads")
EXCEL_FILE_PATH = BASE_UPLOADS_DIR / "input.xlsx"
DISTANCE_MATRIX_PATH = BASE_UPLOADS_DIR / "Distance_matrix.xlsx"
DATA_JSON_PATH = BASE_UPLOADS_DIR / "data.json"
BASE_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# >>> THÊM DÒNG NÀY ĐỂ PHỤC VỤ FILE TĨNH TỪ THƯ MỤC UPLOADS <<<
# Client sẽ truy cập qua http://127.0.0.1:8000/Uploads/input.xlsx
app.mount("/Uploads", StaticFiles(directory=BASE_UPLOADS_DIR), name="Uploads")

# Biến trạng thái xử lý
processing_status = {"status": "idle", "message": ""}

# # Thử kết nối Backend
# @app.get("/")
# async def read_root():
#     return {"Hello": "World"}

# Endpoint upload excel vẫn có thể giữ lại nếu bạn muốn có cách cập nhật input.xlsx thủ công
@app.post("/api/upload-excel/")
async def upload_excel_file(file: UploadFile = File(...)):
    try:
        if EXCEL_FILE_PATH.exists():
            EXCEL_FILE_PATH.unlink()
        with open(EXCEL_FILE_PATH, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        return {"message": f"Tệp '{file.filename}' đã được tải lên và lưu tại '{EXCEL_FILE_PATH}' thành công."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Không thể lưu tệp Excel: {str(e)}")
    finally:
        if file: # Đảm bảo file tồn tại trước khi đóng
            await file.close()



@app.post("/api/process-vrp/")
async def process_vrp(background_tasks: BackgroundTasks):
    global processing_status
    if processing_status["status"] == "processing":
        raise HTTPException(status_code=400, detail="Hệ thống đang xử lý một yêu cầu khác. Vui lòng đợi.")
    
    if not EXCEL_FILE_PATH.exists():
        raise HTTPException(status_code=400, detail="Vui lòng tải file Excel lên trước khi xử lý.")

    # Đặt trạng thái xử lý
    processing_status = {"status": "processing", "message": "Đang tính toán ma trận khoảng cách..."}
    
    def run_vrp_processing():
        global processing_status
        try:
            # Bước 1: Tính toán ma trận khoảng cách
            calculate_distance_matrix()
            processing_status["message"] = "Đang huấn luyện mô hình AI..."
            
            # Bước 2: Huấn luyện mô hình
            train_vrp_model()
            processing_status["message"] = "Đang chạy suy luận để tạo tuyến đường..."
            
            # Bước 3: Chạy suy luận
            run_vrp_inference()
            processing_status = {"status": "completed", "message": "Xử lý hoàn tất! Kết quả đã được lưu vào data.json."}
        except Exception as e:
            processing_status = {"status": "error", "message": f"Lỗi khi xử lý: {str(e)}"}
            raise e

    # Chạy xử lý trong background để không chặn response
    background_tasks.add_task(run_vrp_processing)
    return {"message": "Bắt đầu xử lý VRP. Kiểm tra trạng thái qua /api/processing-status/"}

@app.get("/api/processing-status/")
async def get_processing_status():
    return processing_status


@app.get("/api/get-routes-from-json/")
async def get_routes_from_json():
    if not DATA_JSON_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Tệp '{DATA_JSON_PATH}' không tìm thấy trên server.")
    try:
        with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
            routes_data = json.load(f)
        return routes_data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Tệp '{DATA_JSON_PATH}' không phải là JSON hợp lệ.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi đọc tệp JSON: {str(e)}")

# ... (phần còn lại của main.py nếu có)

# Để chạy ví dụ này:
# 1. Cài đặt: pip install fastapi uvicorn python-multipart
# 2. Tạo thư mục "Uploads" cùng cấp với file main.py
# 3. Đặt file "data.json" của bạn vào trong thư mục "Uploads".
#    Ví dụ nội dung data.json: [[0,1,2,3,0],[0,5,4,0]]
# 4. Chạy server:  uvicorn main:app --reload --port 8000  
#    (frontend của bạn đang gọi http://127.0.0.1:8000)