from src.kitti_loader import KITTILoader
from src.visualize import show_point_cloud

def main():
    sequence_path = "/data/datasets/kitti/sequences/00"
    loader = KITTILoader(sequence_path)

    points = loader.get_point_cloud(0)
    show_point_cloud(points)












if __name__ == "__main__":
    main()