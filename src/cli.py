import argparse
from src.guided import run_end_to_end
from src.insight import analyze_and_export
from src.powerbi import generate_pbids
from src.ml import train_value_regressor

def main():
    p = argparse.ArgumentParser(prog="analytics-studio")
    sub = p.add_subparsers(dest="cmd")

    a = sub.add_parser("analyze")
    a.add_argument("--file")
    a.add_argument("--table")
    a.add_argument("--save-table")

    w = sub.add_parser("wizard")
    w.add_argument("--file")
    w.add_argument("--table")
    w.add_argument("--save-table")
    w.add_argument("--force", action="store_true")

    b = sub.add_parser("pbids")
    b.add_argument("--table", required=True)

    m = sub.add_parser("train")
    m.add_argument("--file")
    m.add_argument("--table")

    args = p.parse_args()
    if args.cmd == "analyze":
        print(analyze_and_export(file_path=args.file, table_name=args.table, save_clean_table=args.save_table))
    elif args.cmd == "wizard":
        print(run_end_to_end(file_path=args.file, table_name=args.table, save_clean_table=args.save_table, force=args.force))
    elif args.cmd == "pbids":
        print(generate_pbids(args.table))
    elif args.cmd == "train":
        print(train_value_regressor(file_path=args.file, table_name=args.table))
    else:
        p.print_help()

if __name__ == "__main__":
    main()
