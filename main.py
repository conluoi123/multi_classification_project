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

def main():
    parser = argparse.ArgumentParser(description="Master CLI Runner - Đồ án 2 Phân loại Đa lớp (Nhóm 04)")
    parser.add_argument('--module', type=str, choices=['binary_reductions', 'direct_multiclass'],
                        help="Chạy toàn bộ thực nghiệm của một nhóm chiến lược.")
    parser.add_argument('--experiment', type=str, 
                        choices=['visual', 'ovo', 'performance', 'calibration', 'complexity', 'tree', 'svm', 'adaboost'],
                        help="Chạy một kịch bản thực nghiệm cụ thể.")
    parser.add_argument('--all', action='store_true', help="Chạy toàn bộ tất cả thực nghiệm của dự án.")
    
    args = parser.parse_args()
    
    if not (args.module or args.experiment or args.all):
        parser.print_help()
        print("\n Ví dụ cách chạy:")
        print("  python main.py --all")
        print("  python main.py --module binary_reductions")
        print("  python main.py --experiment tree")
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
        print(" BẮT ĐẦU CHẠY THỰC NGHIỆM NHÓM 2: DIRECT MULTI-CLASS")
        print("="*70)
        run_decision_tree_experiment()
        run_multiclass_svm_experiment()
        run_adaboost_mh_experiment()
        
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
        
    print(f"\n HOÀN TẤT THỰC NGHIỆM TRONG {time.time() - start_time:.2f} GIÂY.")
    print(f" Toàn bộ biểu đồ kết quả đã được lưu tại thư mục 'figures/'.")

if __name__ == "__main__":
    main()
