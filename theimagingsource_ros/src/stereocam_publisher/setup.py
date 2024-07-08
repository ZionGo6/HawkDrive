from setuptools import setup
import os
from glob import glob

package_name = 'stereocam_publisher'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch/*.py'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config/*.yaml'))),
    
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='a',
    maintainer_email='slxx5237@gmail.com',
    description='stereo camera publisher',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'leftcamera_publisher = stereocam_publisher.stereocam_publisher:main',
        'rightcamera_publisher = stereocam_publisher.stereocam_publisher:main',
        'msg_sync_subscriber = stereocam_publisher.msg_sync_subscriber:main',
        ],
    }
)
