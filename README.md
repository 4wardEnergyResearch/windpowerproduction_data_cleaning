<a id="readme-top"></a>
<!--
Adapted from the Best-README-Template available at
https://github.com/othneildrew/Best-README-Template/
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://www.4wardenergy.at/de/referenzen/ai4wind">
    <img src="https://www.4wardenergy.at/fileadmin/user_upload/AI4Wind_Logo_400x400-01.jpg" alt="Logo" width="300" height="300">
  </a>

<h3 align="center">Wind power production data cleaning routine</h3>

  <p align="center">
    A versatile 4-step data cleaning routine for wind power production data
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-this-repository">About This Repository</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About This Repository
This repository contains a data cleaning routine for wind power production data that was developed in the course of the [AI4Wind](https://www.4wardenergy.at/de/referenzen/ai4wind) project. It consists of four consecutive stages that can be tuned individually to the specific requirements of the turbines to be analyzed. The routine is designed with batch processing in mind and can be adapted to handle large numbers of turbines with varying turbine types.

### Input data
#### Time series
The input time series data (`example_data/input_time_series/`) is expected to be in the form of a delimited file with at least a time stamp, a wind direction, and a windspeed column. For specifics on the column names, refer to line 95 and following in `src/data_cleaning/timeseries_cleaning.py`.
#### Turbine metadata
Metadata (`example_data/input_metadata`) must be provided in the form of an SQLite database entry. See the exemplary database at `example_data/input_metadata/turbine_metadata.db` for the expected structure. Also, a data cleaning confiquration must be provided for each turbine in the `example_data/input_metadata/data_cleaning_config.xlsx` file.

### Data cleaning
After resampling the input data to 10 min intervals, the main data cleaning routine is executed. It consists of four stages:

1. **Flag Invalid Data:**
   The `flag_invalid_data` function is the first stage where initial validation of the data occurs. It is the only mandatory stage in the routine and has no options available in the config file. This stage introduces a "Flag Computed" column in the data frame to indicate invalid data points based on specific criteria:
   - **Univariate extreme values:** Flags wind speed values below 0 m/s or above 40 m/s.
   - **Power values:** Flags power values that are significantly out of the expected range, specifically less than -5% or more than 105% of the nominal power.
   - **Zero power for high wind speeds:** Flags instances where power is near zero for wind speeds more than twice the cut-in speed.

2. **Power curve shifting:**
   The `compute_flag_pc_shift` function applies a method to identify and flag data points by shifting the power curve "to the left and right" (wind speed) as well as "up and down" (power). This method is based on the assumption that the actual power output should be close to the expected values from the power curve, adjusted by certain thresholds:
   - **Power Curve Shifting:** The manufacturer’s power curve is shifted by a certain percentage of the rated wind speed and nominal power to create conservative bounds. Data points falling outside these shifted bounds are flagged.
   - **Thresholding:** Flags data where the power is above the left-shifted power curve and below the right-shifted power curve, ensuring the flagged data points are those that significantly deviate from expected performance.

3. **Sigma Clipping:**
   The `compute_flag_sigma` function uses sigma clipping to flag outliers based on statistical deviation. This method involves binning the data and calculating statistical metrics to identify anomalies:
   - **Binning and Medians Calculation:** The data is binned by power output, and for each bin, the median and standard deviation of wind speed are calculated.
   - **Filtering and Smoothing:** The calculated medians and standard deviations are smoothed using Savitzky-Golay filters to create robust thresholds.
   - **Sigma Clipping:** After smoothing, data points outside a specified number of standard deviations (sigma) from the median are flagged as outliers. This approach helps to remove data points that are statistically unlikely given the rest of the dataset.

4. **Quantile Mapping:**
   The `compute_flag_quantiles` function is based on [a data cleaning routine by ggoretti](https://github.com/ggoretti/data_cleaning), which applies quantile-based flagging, which is useful for identifying and removing extreme values that deviate from expected trends based on quantiles. It is particularly useful for filtering out horizontal ridges in the data caused by throttling. Since the exemplary data set does not contain these, the routine is deactivated by default. The original method is also included in the code, but cannot be activated in the config file.
   - **Quantile Calculation:** Wind speed data is grouped, and quantiles are calculated for each group to create a detailed profile of the data distribution.
   - **Derivative Check:** The derivative of the quantiles is checked to identify points where the power output decreases significantly, indicating potential outliers.
   - **Thresholding and Masking:** Data points are flagged based on these quantile-derived thresholds, with a focus on ensuring that the flagging adapts to changes in the data distribution.

Stages 2,3 and 4 can be adjusted for different turbine types by changing the parameters in the `example_data/input_metadata/data_cleaning_config.xlsx` file. The columns are annotated using Excel comments.

The data cleaning routine outputs flagged CSV files in the `example_data/output_flagged_time_series/` folder. It includes a "Flag Computed" column that indicates the reason for flagging each data point:

| Flag | Description |
| --- | --- |
| 0 | No flag |
| 1 | Invalid data |
| 2 | Quantile mapping (our method) |
| 2, 3, 4 | Quantile mapping (Reserved for ggoretti's original method) |
| 5 | Sigma clipping |
| 6 | Power curve shifting |


### Diagnostics
After the data cleaning routine has processed all time series in the input folder, diagnostics are computed for the flagged time series and saved in `output_diagnostics`:
- A diagnostic graph is output for each turbine. It shows the data points, color-coded for each Flag Computed value, as well as a histogram of the flagged data points.
- `Flag Evaluation Results.xlsx` contains a summary of the flagged data points for each turbine with some statistics. Turbines which have more than 10% of their data flagged are marked in red. Clicking the link in the "File Link" column opens a diagnostic graph for the respective turbine.
- The `SuspiciousTurbines` folder contains diagnostic graphs for the red-marked turbines.




<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started
To get a local copy up and running, follow these simple steps:


### Prerequisites
- **Python:** The project is developed in Python. Ensure you have Python 3.7 or later installed on your system. You can download it from [python.org](https://www.python.org).
- **pip:** Ensure that Python's package manager, pip, is installed. pip usually comes with Python, but if you need to install it, follow the instructions at [pip.pypa.io](https://pip.pypa.io).

### Installation

#### 1. Clone the Repository
First, you need to clone the repository from GitHub to your local machine. Open your command line interface (CLI) and run the following command:
```bash
git clone --depth=1 https://github.com/4wardEnergyResearch/windpowerproduction_data_cleaning.git
cd windpowerproduction_data_cleaning
```
#### 2. Set Up a Virtual Environment (Recommended)
It's recommended to use a Python virtual environment to avoid conflicts with other packages and manage dependencies efficiently. If you don't have virtualenv installed, you can install it using pip:
```bash
pip install virtualenv
```
Create and activate a virtual environment in the project directory:
##### For Windows
```bash
python -m venv env
env\Scripts\activate
```
##### For macOS and Linux
```bash
python3 -m venv env
source env/bin/activate
```

#### 3. Install dependencies
Once the virtual environment is activated, install the project dependencies using:
```bash
pip install -r requirements.txt
```

#### 4. Verify installation
To verify that the installation was successful, you can run the `example.py` script:
```bash
python example.py
```
If there are no errors and the program runs through, your installation is complete.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html.en)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact
Stefan Janisch - [stefan.janisch@4wardenergy.at](mailto:stefan.janisch@4wardenergy.at)

[Project website](https://www.4wardenergy.at/en/references/lowtemp4districtheat)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/4wardEnergyResearch/windpowerproduction_data_cleaning.svg?style=for-the-badge
[contributors-url]: https://github.com/4wardEnergyResearch/windpowerproduction_data_cleaning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/4wardEnergyResearch/windpowerproduction_data_cleaning.svg?style=for-the-badge
[forks-url]: https://github.com/4wardEnergyResearch/windpowerproduction_data_cleaning/network/members
[stars-shield]: https://img.shields.io/github/stars/4wardEnergyResearch/windpowerproduction_data_cleaning.svg?style=for-the-badge
[stars-url]: https://github.com/4wardEnergyResearch/windpowerproduction_data_cleaning/stargazers
[issues-shield]: https://img.shields.io/github/issues/4wardEnergyResearch/windpowerproduction_data_cleaning.svg?style=for-the-badge
[issues-url]: https://github.com/4wardEnergyResearch/windpowerproduction_data_cleaning/issues
[license-shield]: https://img.shields.io/github/license/4wardEnergyResearch/windpowerproduction_data_cleaning.svg?style=for-the-badge
[license-url]: https://github.com/4wardEnergyResearch/windpowerproduction_data_cleaning/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/4ward-energy-research-gmbh/
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
