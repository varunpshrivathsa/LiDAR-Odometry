from src.kitti_loader import KITTILoader

def main():
    sequence_path = "/data/datasets/kitti/sequences/00"
    loader = KITTILoader(sequence_path)

    print("Total Scans:",len(loader))
    points = loader.get_point_cloud(0)
    print("First Scan Shape: ",points.shape)
    print(points[:5])













if __name__ == "__main__":
    main()