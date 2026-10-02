from setuptools import find_packages, setup

import os
from glob import glob

package_name = 'ssf_gazebo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Gazebo simulation package for navigation with mobile robots',
    license='MIT',
    entry_points={
        'console_scripts': [
            'odom_tf_broadcaster = ssf_gazebo.odom_tf_broadcaster:main',
            'rollout_data_generator = ssf_gazebo.rollout_data_generator:main',
        ],
    },
)
