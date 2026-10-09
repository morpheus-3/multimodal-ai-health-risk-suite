import argparse
import json
from . import health

def main():
    parser = argparse.ArgumentParser(description="Train Health Risk Lab")
    parser.add_argument("data")
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--synthetic",action="store_true")
    parser.add_argument("--framework",choices=["tensorflow","pytorch"],default="tensorflow")
    parser.add_argument("--ratings")
    args=parser.parse_args()
    result=health.train(args.data,args.epochs,args.synthetic)
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
