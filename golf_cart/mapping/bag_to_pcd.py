import sys
import os
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import sensor_msgs.msg
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

def save_pcd(cloud_msg, file_name):
    """Converts a PointCloud2 message to a PCD file."""
    # Convert to numpy (x, y, z, intensity if available)
    field_names = [f.name for f in cloud_msg.fields]
    # Common fields to look for
    target_fields = ['x', 'y', 'z']
    if 'intensity' in field_names:
        target_fields.append('intensity')
    elif 'i' in field_names:
        target_fields.append('i')

    # Read points
    points = list(pc2.read_points(cloud_msg, field_names=target_fields, skip_nans=True))

    if not points:
        print(f"Warning: Empty cloud for {file_name}")
        return

    with open(file_name, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write(f"FIELDS {' '.join(target_fields)}\n")
        f.write(f"SIZE {' '.join(['4'] * len(target_fields))}\n")
        f.write(f"TYPE {' '.join(['F'] * len(target_fields))}\n")
        f.write(f"COUNT {' '.join(['1'] * len(target_fields))}\n")
        f.write(f"WIDTH {len(points)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(points)}\n")
        f.write("DATA ascii\n")

        for p in points:
            f.write(' '.join(map(str, p)) + "\n")

def main():
    if len(sys.argv) < 4:
        print("Usage: python3 bag_to_pcd.py <bag_path> <topic> <output_dir>")
        return

    bag_path = sys.argv[1]
    topic_name = sys.argv[2]
    output_dir = sys.argv[3]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr')

    reader = rosbag2_py.SequentialReader()
    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"Error opening bag: {e}")
        return

    storage_filter = rosbag2_py.StorageFilter(topics=[topic_name])
    reader.set_filter(storage_filter)

    count = 0
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg = deserialize_message(data, sensor_msgs.msg.PointCloud2)

        # Save pcd
        # Use timestamp for filename
        timestamp = t
        file_name = os.path.join(output_dir, f"{timestamp}.pcd")

        save_pcd(msg, file_name)
        count += 1
        if count % 10 == 0:
            print(f"Processed {count} frames...")

    print(f"Done! Saved {count} PCD files to {output_dir}")

if __name__ == "__main__":
    main()
