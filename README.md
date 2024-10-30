
# Code for "Causal discovery in the presence of endogenous context variables"
Note: this code is for reviewing purposes only. An official version will be released upon acceptance.

## Description

This is the code for the simulation study of the paper, including all algorithms that were used in the study: PC-AR, PC-B, PC-P, PC-M, JCI-FCI and CD-NOD. For further information and references, please consult the main paper.

## Getting started

### Dependencies 

To install the dependencies required for this setup, use the ```requirements.txt``` file as such:

```pip install -r requirements.txt```

## Running the simulation study

### Getting experimental results
There are two main files to be used for running the simulation study, which can be found in the folder ```endogeneous_context_simulation_study```:

1.  runs the experiments for PC-AR, PC-M, PC-P and PC-B.
2. ```compute_fci_non_timeseries.py``` runs the experiments for JCI-FCI and CD-NOD. 

All other code files necessary for the experimental results can also be found there. 

We use YAML files to store the experimental setup configurations. Make sure you change the ```save_folder``` according to your needs. The YAML files that have been used to generate the results in the paper can be found in the folder  ```update_configs```. 

As described in the main paper, we ran our experiments on a cluster. To run these experiments using slurm, run the following scripts as follows:
1. ```python submit_tasks.py PATH_TO_YAML`` for starting jobs with the experiments for  PC-AR, PC-M, PC-P and PC-B.
2. ```python submit_fci_tasks.py PATH_TO_YAML``` for starting jobs with the experiments for  JCI-FCI and CD-NOD. 


Before starting, make sure you change the account name and other parameters in the corresponding submit scripts! Also, make sure ```sequential = False``` is set in ```compute_regime_graphs_non_timeseries.py```and ```compute_fci_non_timeseries.py```. 

If you would like to run the experiments in sequential mode, there are two things to take care of: 
1. set```sequential = True``` in  ```compute_regime_graphs_non_timeseries.py```and ```compute_fci_non_timeseries.py```. 
2. Uncomment the main part in ```compute_regime_graphs_non_timeseries.py```and ```compute_fci_non_timeseries.py```
3. Run the scripts as follows :
	*  ```python compute_regime_graphs_non_timeseries.py PATH_TO_YAML```
	* ```python compute_fci_non_timeseries.py PATH_TO_YAML```
 
### Obtaining the metrics for the experiments
 The results of the experiments are evaluated using the ```metrics.py``` and ```metrics_fci.py``` scripts. To generate the results, run the following commands:
 1. ```python metrics.py PATH_TO_YAML FOLDER_TO_SAVE_METRICS```
 2. ```python metrics_fci.py PATH_TO_YAML FOLDER_TO_SAVE_METRICS```
where ```FOLDER_TO_SAVE_METRICS``` is the folder where the metric files should be saved. 

### Obtaining the plots 
The plots are generated using the ```plot_fci.py``` script. To generate plots for a set of configurations, use the following command:
1. ``` python plot_fci.py --yaml_path PATH_TO_YAML --metrics_folder PATH_TO_METRICS_FOLDER --plot_known KNOWN_TYPE --plot_avg PLOT_AVG --metrics_list METRICS_LIST``` for experiments with  PC-AR, PC-M, PC-P and PC-B.
2. ``` python plot_fci.py --yaml_path PATH_TO_YAML --metrics_folder PATH_TO_METRICS_FOLDER --metrics_folder_fci PATH_TO_FCI_CDNOD_METRICS_FOLDER --plot_known KNOWN_TYPE --plot_avg PLOT_AVG --metrics_list METRICS_LIST --plot_fci True```
where the following arguments must be set (besides the already mentioned ```--yaml_path``` and ```--metrics_folder```)
```--plot_avg``` sets whether you would like to plot the metric values averaged over contexts, or for all contexts individually. For the plots in the paper, this is set to ```True```
```--metrics_list``` sets which metrics should be plotted. In the paper we use ```tpr,fpr``` for the TPR and FPR
```--plot_fci``` sets whether the plots should include the metrics for FCI and CD-NOD. If set to ```True```, then ```--metrics_folder_fci``` must be set to the path where the metrics for the FCI and CD-NOD are. 
```--plot_known``` a list of configs to plot that indicate if the regime is known, i.e., whether all links should be evaluated for the metrics (False), whether the links from regime to its children should be ignored (True)
        or whether the links to and from the regime indicator should be ignored ('and_parents'). Example: ```--plot_known and_parents,True,False``` or ```--plot_known True,False``` 

## Further details

We also submit our metric files, which can be found in the ```metrics_update``` folder
As discussed in the paper, not all experiment trials are succesful. We save, for each configuration, how many trials have failed  in the ```failed.txt``` file in the ```metrics_update``` folder. 

## License
Since we are building on tigramite, we use the GNU General Public License as published by the Free Software Foundation; version 3 of the License or later. Here is a list of all packages and their licenses: 


| Package                    | Version      | License       | Description                                                      |
|----------------------------|--------------|---------------|------------------------------------------------------------------|
| anyio                      | 3.5.0        | MIT           | Compatibility layer for multiple asynchronous event loop implementations |
| argon2-cffi                | 21.3.0       | MIT           | Password hashing using the Argon2 algorithm                      |
| argon2-cffi-bindings       | 21.2.0       | MIT           | Low-level CFFI bindings for Argon2                               |
| asttokens                  | 2.0.5        | Apache-2.0    | Annotate Python AST with the source code that generated it       |
| async-lru                  | 2.0.4        | MIT           | Simple LRU cache for asyncio                                     |
| attrs                      | 23.1.0       | MIT           | Classes without boilerplate                                      |
| Babel                      | 2.11.0       | BSD           | Internationalization and localization library                    |
| backcall                   | 0.2.0        | BSD           | Specifications for callback functions passed in to an API        |
| beautifulsoup4             | 4.12.2       | MIT           | Library for parsing HTML and XML documents                       |
| bleach                     | 4.1.0        | Apache-2.0    | HTML sanitizing library that prevents XSS attacks                |
| Bottleneck                 | 1.3.5        | BSD           | Fast NumPy array functions optimized for performance             |
| Brotli                     | 1.0.9        | MIT           | Generic-purpose lossless compression algorithm by Google         |
| cairocffi                  | 1.6.1        | BSD           | CFFI-based drop-in replacement for Pycairo                       |
| CairoSVG                   | 2.7.1        | LGPL-3.0      | SVG converter based on Cairo                                    |
| causal-learn               | 0.1.3.8      | MIT           | Library for causal learning algorithms                          |
| certifi                    | 2023.11.17   | MPL-2.0       | Python package for providing Mozilla's CA Bundle                 |
| cffi                       | 1.16.0       | MIT           | Foreign Function Interface for Python to call C code             |
| charset-normalizer         | 2.0.4        | MIT           | Library for character encoding detection                         |
| comm                       | 0.1.2        | BSD           | Communication tools for various protocols                        |
| contourpy                  | 1.2.0        | BSD           | Python wrapper for 2D contour plotting and extraction            |
| cryptography               | 41.0.7       | Apache-2.0    | Cryptographic recipes and primitives for Python                  |
| cssselect2                 | 0.7.0        | BSD           | CSS selectors for Python ElementTree                             |
| cycler                     | 0.12.1       | BSD           | Composable keyword argument processor                            |
| debugpy                    | 1.6.7        | MIT           | Debugger for Python, adaptable for multiple editors              |
| decorator                  | 5.1.1        | BSD           | Simplify making function decorators and wrappers                 |
| defusedxml                 | 0.7.1        | PSFL          | XML bomb protection for Python stdlib modules                    |
| exceptiongroup             | 1.0.4        | MIT           | Backport of the Python 3.11 ExceptionGroup to earlier versions   |
| executing                  | 0.8.3        | MIT           | Library to get the currently executing AST node                  |
| fastjsonschema             | 2.16.2       | BSD           | Fast JSON schema validator for Python                            |
| fonttools                  | 4.47.0       | MIT           | Library to manipulate font files programmatically                |
| graphviz                   | 0.20.3       | MIT           | Simple Python interface for Graphviz                             |
| idna                       | 3.4          | BSD           | Internationalized Domain Names in Applications (IDNA)            |
| ipykernel                  | 6.25.0       | BSD           | IPython Kernel for Jupyter                                       |
| ipython                    | 8.15.0       | BSD           | Powerful interactive Python shell                                |
| jedi                       | 0.18.1       | MIT           | Autocompletion and code analysis library for Python              |
| Jinja2                     | 3.1.2        | BSD           | A modern and designer-friendly templating language for Python    |
| joblib                     | 1.3.2        | BSD           | Lightweight pipelining: using Python functions as pipeline jobs  |
| json5                      | 0.9.6        | MIT           | A modern format for JSON data with ES5 features                  |
| jsonschema                 | 4.19.2       | MIT           | An implementation of JSON Schema for Python                      |
| jsonschema-specifications  | 2023.7.1     | MIT           | Specifications and tools around JSONSchema                       |
| jupyter_client             | 8.6.0        | BSD           | Client for the Jupyter protocol                                  |
| jupyter_core               | 5.5.0        | BSD           | Core common functionality of Jupyter/IPython                     |
| jupyter-events             | 0.8.0        | BSD           | Event dispatching and handling in the Jupyter ecosystem          |
| jupyter-lsp                | 2.2.0        | BSD           | Language Server Protocol integration for Jupyter                 |
| jupyter_server             | 2.10.0       | BSD           | Backend server for Jupyter web applications                      |
| jupyter_server_terminals   | 0.4.4        | BSD           | Terminal functionality for Jupyter Server                        |
| jupyterlab                 | 4.0.8        | BSD           | JupyterLab: the next-generation web-based notebook               |
| jupyterlab-pygments        | 0.1.2        | BSD           | Pygments theme using JupyterLab CSS variables                    |
| jupyterlab_server          | 2.25.1       | BSD           | Server components for JupyterLab and JupyterLab-like applications|
| kiwisolver                 | 1.4.5        | BSD           | A fast implementation of the Cassowary constraint solver         |
| llvmlite                   | 0.39.1       | BSD           | Lightweight LLVM python binding for writing JIT compilers        |
| MarkupSafe                 | 2.1.3        | BSD           | Implements a XML/HTML/XHTML Markup safe string for Python        |
| matplotlib                 | 3.8.2        | PSF           | Python plotting package                                          |
| matplotlib-inline          | 0.1.6        | BSD           | Inline Matplotlib graphics in Jupyter environments               |
| mistune                    | 2.0.4        | BSD           | Fastest markdown parser in pure Python                           |
| mkl-fft                    | 1.3.8        | BSD           | NumPy-based implementation of Fast Fourier Transform using MKL   |
| mkl-random                 | 1.2.4        | BSD           | NumPy-based implementation of Random Number Generation using MKL |
| mkl-service                | 2.4.0        | BSD           | Python wrappers for Intel MKL service functions                  |
| nbclient                   | 0.8.0        | BSD           | A client library for executing notebooks                         |
| nbconvert                  | 7.10.0       | BSD           | Convert Jupyter Notebooks to other formats                       |
| nbformat                   | 5.9.2        | BSD           | Jupyter Notebook format                                           |
| nest-asyncio               | 1.5.6        | BSD           | Patch asyncio to allow nested event loops                        |
| networkx                   | 3.2.1        | BSD           | Create and manipulate complex networks                            |
| notebook_shim              | 0.2.3        | BSD           | Compatibility layer between Jupyter Notebook and its successor    |
| numba                      | 0.56.4       | BSD           | JIT compiler that translates a subset of Python and NumPy code into fast machine code |
| numexpr                    | 2.8.7        | MIT           | Fast numerical expression evaluator for NumPy                    |
| numpy                      | 1.23.5       | BSD           | Scientific computing with Python                                  |
| overrides                  | 7.4.0        | Apache 2.0    | A decorator to automatically override methods                     |
| packaging                  | 23.1         | BSD           | Core utilities for Python packages                                |
| pandas                     | 2.1.4        | BSD           | Powerful data structures for data analysis, time series, and statistics |
| pandocfilters              | 1.5.0        | BSD           | Utilities for writing pandoc filters in python                    |
| parso                      | 0.8.3        | MIT           | Python parser that supports error recovery and round-trip parsing for different Python versions |
| patsy                      | 0.5.6        | BSD           | Describe statistical models and build design matrices             |
| pexpect                    | 4.8.0        | ISC           | Python module for spawning child applications and controlling them |
| pickleshare                | 0.7.5        | MIT           | Small 'shelve'-like database with concurrency support             |
| pillow                     | 10.2.0       | MIT-CMU       | Python Imaging Library (Fork)                                     |
| pip                        | 23.3.1       | MIT           | The Python package installer                                      |
| platformdirs               | 3.10.0       | MIT           | A small Python module for determining appropriate platform-specific dirs |
| prometheus-client          | 0.14.1       | Apache-2.0    | Python client for the Prometheus monitoring system                |
| prompt-toolkit             | 3.0.43       | BSD           | Library for building powerful interactive command lines in Python |
| psutil                     | 5.9.0        | BSD           | Cross-platform lib for process and system monitoring in Python    |
| ptyprocess                 | 0.7.0        | ISC           | Run a subprocess in a pseudo terminal                             |
| pure-eval                  | 0.2.2        | MIT           | Safely evaluate Python expressions from untrusted sources         |
| pycparser                  | 2.21         | BSD           | C parser in Python                                                |
| pydot                      | 2.0.0        | MIT           | Python interface to Graphviz's Dot language                       |
| Pygments                   | 2.15.1       | BSD           | Syntax highlighting package                                       |
| pyOpenSSL                  | 23.2.0       | Apache-2.0    | Python wrapper around a subset of the OpenSSL library's functions |
| pyparsing                  | 3.1.1        | MIT           | General parsing module                                            |
| PySocks                    | 1.7.1        | BSD           | Python SOCKS client module                                        |
| python-dateutil            | 2.8.2        | BSD           | Extensions to the standard Python datetime module                 |
| python-json-logger         | 2.0.7        | BSD           | A python library adding a json log formatter                      |
| pytz                       | 2023.3.post1 | MIT           | World timezone definitions and modern and historical adjustments  |
| PyYAML                     | 6.0.1        | MIT           | YAML parser and emitter for Python                                |
| pyzmq                      | 25.1.0       | BSD           | Python bindings for 0MQ                                           |
| referencing                | 0.30.2       | MIT           | Module for managing references in academic writing                |
| requests                   | 2.31.0       | Apache-2.0    | HTTP library for Python                                           |
| rfc3339-validator          | 0.1.4        | MIT           | Validator for RFC 3339 date-time string                           |
| rfc3986-validator          | 0.1.1        | MIT           | Validator for RFC 3986 URI string                                 |
| rpds-py                    | 0.10.6       | MIT           | Python implementation of recursive persistent data structures     |
| scikit-learn               | 1.4.0rc1     | BSD           | Machine learning library for Python                               |
| SciPy                      | 1.12.0rc1    | BSD           | Scientific computing and technical computing                       |
| Send2Trash                 | 1.8.2        | BSD           | Send files to the Trash or Recycle Bin                            |
| setuptools                 | 68.2.2       | MIT           | Easily download, build, install, upgrade, and uninstall Python packages |
| six                        | 1.16.0       | MIT           | Python 2 and 3 compatibility utilities                            |
| sniffio                    | 1.2.0        | Apache-2.0    | Sniff out which async library your code is running under          |
| soupsieve                  | 2.5          | MIT           | A modern CSS selector implementation for Beautiful Soup           |
| stack-data                 | 0.2.0        | MIT           | Extract data from Python stack frames                             |
| statsmodels                | 0.14.2       | BSD           | Statistical modeling and econometrics in Python                   |
| terminado                  | 0.17.1       | BSD           | Terminals served to xterm.js using Tornado websockets             |
| threadpoolctl              | 3.2.0        | BSD           | Threadpoolctl: Control the threadpools of native libraries        |
| tigramite                  | 5.2.2.3      | GPL-3.0       | Causal discovery for time series datasets                         |
| tinycss2                   | 1.2.1        | BSD           | A low-level CSS parser for Python                                 |
| tomli                      | 2.0.1        | MIT           | A lil' TOML parser                                                |
| tornado                    | 6.3.3        | Apache-2.0    | Python web framework and asynchronous networking library          |
| tqdm                       | 4.66.4       | MPL-2.0, MIT  | Fast, extensible progress bar for loops and iterable              |
| traitlets                  | 5.7.1        | BSD           | Configuration system for Python applications                      |
| typing_extensions          | 4.7.1        | PSF           | Backported and experimental type hints for Python 3.5+            |
| tzdata                     | 2023.3       | Apache 2.0    | Timezone data from the IANA Time Zone Database                    |
| urllib3                    | 1.26.18      | MIT           | HTTP client for Python (used by Requests)                         |
| wcwidth                    | 0.2.5        | MIT           | Measure the displayed width of Unicode strings                    |
| webencodings               | 0.5.1        | BSD           | Character encoding aliases for legacy web content                 |
| websocket-client           | 0.58.0       | LGPL          | WebSocket client for Python                                       |
| wheel                      | 0.41.2       | MIT           | A built-package format for Python                                 |

