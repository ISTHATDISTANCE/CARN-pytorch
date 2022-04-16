import argparse
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)

    return parser.parse_args()

def main():
    cfg = parse_args()
    src = cv2.imread(cfg.src)
    filename = cfg.src.split('/')[-1]
    print(filename)
    # cv2.imwrite(f"./{filename}", src[0: 100, 256:356])
    # cv2.imwrite(f"./{filename}", src[160:260, 10:110])
    cv2.imwrite(f"./{filename}", src[110:210, 150:250])

if __name__ == "__main__":
    main()