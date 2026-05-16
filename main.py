import sys
import os
import argparse
import time

# Thêm thư mục gốc vào sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.experiments.exp_binary_reductions import (
    run_visual_experiments,
    run_ovo_detailed_analysis,
    run_performance_experiments,
    run_calibration_experiments,
    run_complexity_and_stats
)
from src.experiments.exp_direct_multiclass import (
    run_decision_tree_experiment,
    run_multiclass_svm_experiment,
    run_adaboost_mh_experiment
)
from src.experiments.exp_tikz_compiler import compile_tikz_diagrams
from src.experiments.exp_real_datasets import run_mnist_experiment, run_eurlex_experiment
from src.experiments.exp_optimization_benchmark import run_optimized_rich_analysis

def main():
    parser = argparse.ArgumentParser(description="Master CLI Runner - Đồ án 2 Phân loại Đa lớp (Nhóm 04)")
    parser.add_argument('--module', type=str, choices=['binary_reductions', 'direct_multiclass', 'theory', 'real_datasets', 'optimization_benchmark'],
                        help="Chạy toàn bộ thực nghiệm của một nhóm chiến lược, biên dịch TikZ, dữ liệu thực tế hoặc tối ưu hóa.")
    parser.add_argument('--experiment', type=str, 
                        choices=['visual', 'ovo', 'performance', 'calibration', 'complexity', 'tree', 'svm', 'adaboost', 'tikz', 'mnist', 'eurlex', 'optimize'],
                        help="Chạy một kịch bản thực nghiệm cụ thể.")
    parser.add_argument('--all', action='store_true', help="Chạy toàn bộ tất cả thực nghiệm của dự án.")
    parser.add_argument('--tikz', action='store_true', help="Biên dịch riêng các biểu đồ lý thuyết TikZ (LaTeX) vào figures/theory.")
    parser.add_argument('--optimize', action='store_true', help="Chạy bộ phân tích toàn diện (Rich Analysis) sử dụng các mô hình tối ưu hóa siêu tốc.")
    parser.add_argument('--pure-real', action='store_true', help="Chạy thực nghiệm trên dữ liệu thực tế bằng các thuật toán gốc chưa tối ưu hóa (đơn luồng/chậm hơn).")
    
    args = parser.parse_args()
    
    if not (args.module or args.experiment or args.all or args.tikz or args.optimize or args.pure_real):
        parser.print_help()
        print("\n Ví dụ cách chạy:")
        print("  python main.py --all")
        print("  python main.py --optimize")
        print("  python main.py --module real_datasets")
        print("  python main.py --experiment mnist")
        print("  python main.py --experiment eurlex")
        print("  python main.py --tikz")
        sys.exit(0)
        
    start_time = time.time()
    
    if args.all or args.module == 'binary_reductions':
        print("\n" + "="*70)
        print(" BẮT ĐẦU CHẠY THỰC NGHIỆM NHÓM 1: BINARY REDUCTIONS")
        print("="*70)
        run_visual_experiments()
        run_ovo_detailed_analysis()
        run_performance_experiments()
        run_calibration_experiments()
        run_complexity_and_stats()
        
    if args.all or args.module == 'direct_multiclass':
        print("\n" + "="*70)
        print("🚀 BẮT ĐẦU CHẠY THỰC NGHIỆM NHÓM 2: DIRECT MULTI-CLASS")
        print("="*70)
        run_decision_tree_experiment()
        run_multiclass_svm_experiment()
        run_adaboost_mh_experiment()
        
    if args.all or args.module == 'real_datasets' or args.pure_real:
        print("\n" + "="*70)
        mode_label = "GỐC CHƯA TỐI ƯU" if args.pure_real else "TỐI ƯU HÓA"
        print(f"🚀 BẮT ĐẦU CHẠY THỰC NGHIỆM TRÊN DỮ LIỆU THỰC TẾ ({mode_label})")
        print("="*70)
        use_opt = not args.pure_real
        run_mnist_experiment(use_optimization=use_opt)
        run_eurlex_experiment(use_optimization=use_opt)

    if args.all or args.module == 'optimization_benchmark' or args.optimize or args.experiment == 'optimize':
        run_optimized_rich_analysis()
        
    if args.all or args.module == 'theory' or args.tikz or args.experiment == 'tikz':
        compile_tikz_diagrams()
        
    if args.experiment:
        print("\n" + "="*70)
        print(f" BẮT ĐẦU CHẠY THỰC NGHIỆM: {args.experiment.upper()}")
        print("="*70)
        if args.experiment == 'visual': run_visual_experiments()
        elif args.experiment == 'ovo': run_ovo_detailed_analysis()
        elif args.experiment == 'performance': run_performance_experiments()
        elif args.experiment == 'calibration': run_calibration_experiments()
        elif args.experiment == 'complexity': run_complexity_and_stats()
        elif args.experiment == 'tree': run_decision_tree_experiment()
        elif args.experiment == 'svm': run_multiclass_svm_experiment()
        elif args.experiment == 'adaboost': run_adaboost_mh_experiment()
        elif args.experiment == 'mnist': run_mnist_experiment(use_optimization=not args.pure_real)
        elif args.experiment == 'eurlex': run_eurlex_experiment(use_optimization=not args.pure_real)
        
    print(f"\n HOÀN TẤT THỰC NGHIỆM TRONG {time.time() - start_time:.2f} GIÂY.")
    print(f" Toàn bộ biểu đồ kết quả đã được lưu tại thư mục 'figures/'.")

if __name__ == "__main__":
    main()
