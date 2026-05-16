import os
import subprocess
import shutil

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LATEX_DIR = os.path.join(BASE_DIR, 'latex')
FIG_DIR = os.path.join(BASE_DIR, 'figures', 'theory')

def compile_tikz_diagrams():
    print("\n" + "="*70)
    print(" BẮT ĐẦU BIÊN DỊCH CÁC BIỂU ĐỒ TIKZ (LATEX) VÀO THỰC ĐƠN FIGURES")
    print("="*70)
    
    if not os.path.exists(LATEX_DIR):
        print(f" Không tìm thấy thư mục {LATEX_DIR}")
        return

    os.makedirs(FIG_DIR, exist_ok=True)
    
    # Kiểm tra pdflatex có tồn tại trong PATH không
    if shutil.which('pdflatex') is None:
        print(" CẢNH BÁO: Không tìm thấy 'pdflatex' trong hệ thống (PATH).")
        print(" Để Python tự động dịch file .tex sang .pdf vào figures, máy bạn cần cài đặt MiKTeX hoặc TeX Live và đưa vào PATH.")
        print(f" Mã nguồn TikZ gốc vẫn được lưu trữ an toàn tại '{LATEX_DIR}'. Bạn có thể nhúng trực tiếp lên Overleaf!")
        return

    tex_files = [f for f in os.listdir(LATEX_DIR) if f.endswith('.tex')]
    
    for tex_file in tex_files:
        file_path = os.path.join(LATEX_DIR, tex_file)
        print(f"⏳ Đang dịch {tex_file} -> {FIG_DIR} ...")
        try:
            # Chạy pdflatex với output directory trỏ thẳng vào figures/theory
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', f'-output-directory={FIG_DIR}', file_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace'
            )
            if result.returncode == 0:
                print(f"  → Thành công! Đã xuất {tex_file.replace('.tex', '.pdf')}")
                # Xóa các file rác sinh ra bởi LaTeX (.aux, .log)
                for ext in ['.aux', '.log']:
                    trash = os.path.join(FIG_DIR, tex_file.replace('.tex', ext))
                    if os.path.exists(trash):
                        os.remove(trash)
            else:
                print(f"  →  Lỗi khi dịch {tex_file}. Vui lòng kiểm tra cú pháp LaTeX hoặc log.")
        except Exception as e:
            print(f"  →  Lỗi hệ thống: {e}")
            
    print(f"\n  Hoàn tất quá trình biên dịch TikZ. Các file PDF lý thuyết đã nằm gọn trong '{FIG_DIR}'.\n")

if __name__ == "__main__":
    compile_tikz_diagrams()
