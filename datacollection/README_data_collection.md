# Follow this doc to collect your own data with Aria Glasses
It's for Aria Gen1 glasses for now. The code for Aria Gen2 is coming soon...

## Install Aria Mobile App
Follow [ARK SW Downloads and Updates](https://facebookresearch.github.io/projectaria_tools/docs/ARK/mobile_companion_app). Install the app, sign in and pair.

The Email: Username + @tfbnw.net

## Add Environment Variables
```
vim ~/.bashrc
```

Add the following environment variables to your ~/.bashrc or ~/.zshrc with your institution's username and password

```
export ARIA_MPS_UNAME="YOUR_UNAME"
export ARIA_MPS_PASSW="YOUR_PASSW"
```

```
source ~/.bashrc
```

## Install Aria Studio
```
pip install projectaria_tools==1.7.1
pip install aria_studio==1.1.1 --no-cache-dir
```

Connect your Aria glasses with your computer via USB and run:
```
aria_studio --port 8080
```
The you can go to http://127.0.0.1:8080 via the browser to visit the Aria Studio

## Verify that your dependencies have been installed correctly.
```
pip install projectaria_client_sdk==1.1.0 --no-cache-dir
aria-doctor
```

You should see:
```
[ pass] glibc version ok
[ pass] Python version ok
[ pass] Aria udev rules ok
[ pass] Aria network manager connection ok
```

## Pair the glasses via USB to your computer. Verify that you have connected correctly.
```
aria auth pair
```
You should see:
```
[AriaCli:App][INFO]: Attempting to send authentication pairing request to device over USB. Please ensure the device is connected to a USB port.
[AriaCli:App][INFO]: Sent authentication request with hash xxx to device. Please check and approve the request in the companion app.
```
Approve the request in the app.

## Try to record via the phone
You could customize the profile you want. We recommend this settings:
```
RGB: 30fbs 2MP
SLAM: 30fps VGA
ET: 10fps QVGA
IMUS, Mag, Baro, Audio, GPS, Wi-Fi, BLE: on
```

After recording, you should connect the glasses to your computer and run ```aria_studio```, download the data you just recorded to ```./data/```, which should be a ```.vrs``` file. 

## Install Projectaria Tools and Client SDK
```
pip install projectaria_tools==1.7.1
```

## Data processing on the MPS server
Submit the video for data processing on the MPS server and reorganize the output folder. Job submission may take anywhere from 5 to 30 minutes. For example, if your ```.vrs``` file is ```TEST.vrs```, you would run

```
conda activate humanego
cd data/
NAME="TEST"
aria_mps single --force -i "${NAME}.vrs" -u "$ARIA_MPS_UNAME" -p "$ARIA_MPS_PASSW" --features SLAM HAND_TRACKING
mv "${NAME}.vrs" "mps_${NAME}_vrs/sample.vrs"
mkdir -p "mps_${NAME}_vrs/else/"
mv "${NAME}.vrs.json" "mps_${NAME}_vrs/else/sample.vrs.json"
mv "mps_${NAME}_vrs/vrs_health_check.json" "mps_${NAME}_vrs/else/vrs_health_check.json"
mv "mps_${NAME}_vrs/vrs_health_check_slam.json" "mps_${NAME}_vrs/else/vrs_health_check_slam.json"
```

For an 80-second data, it usually takes around 20-25 minutes to process a single VRS file depending on network conditions and server load.
For a 25-second data, it usually takes around 10 minutes.

After this, it should be:
```
- data
    - mps_TEST_vrs/
        - else
            - sample.vrs.json
            - vrs_health_check.json
            - vrs_health_check_slam.json
        - hand_tracking
            - hand_tracking_results.csv
            - summary.json
        - slam
            - closed_loop_trajectory.csv
            - online_calibration.jsonl
            - open_loop_trajectory.csv
            - semidense_observations.csv.gz
            - semidense_points.csv.gz
            - summary.json
        - sample.vrs
```

## Visualize the aria sensors
```
viewer_aria_sensors --vrs "./test_data/mps_TEST_vrs/sample.vrs"
```

## Visualize the hand tracking and slam
```
viewer_mps --vrs "./test_data/mps_TEST_vrs/sample.vrs" \
--trajectory "./test_data/mps_TEST_vrs/slam/closed_loop_trajectory.csv" \
--points "./test_data/mps_TEST_vrs//slam/semidense_points.csv.gz" \
--hands_all "./test_data/mps_TEST_vrs/hand_tracking/hand_tracking_results.csv" \
--web
```