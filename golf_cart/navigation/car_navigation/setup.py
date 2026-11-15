from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'car_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'bt_trees'), glob('bt_trees/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SeDriCa',
    maintainer_email='sedrica@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'twist_to_control = car_navigation.twist_to_control:main',
            'odom_pub = car_navigation.odometry:main',
            # 'simple_lidar_to_scan = car_navigation.simple_lidar_to_scan:main',
        ],
    },
)
