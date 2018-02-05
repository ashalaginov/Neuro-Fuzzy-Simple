## Neuro-Fuzzy-Simple

Simple implementation of Neuro-Fuzzy algorithm in C++. The program demonstrates how Neuro-Fuzzy architecture works: 
starting from assigning linguistic terms, building fuzzy patches and ending up with generating classification model.

## Fuzzy Patches generation

For each linguistic term we use simple statistical approach without prior knowledge about the dataset. 
For 3-terms fuzzy set "three-sigma rule of thumb" is used to generate 3 linguistic terms: {Low, Medium, High}. For 5-terms fuzzy set corresponding 5 standard deviations are used.
Finally, simple rectangular fuzzy patches are generated.

## Original Paper

You can find more information about the practical experiments and datasets in the following conference paper:

    @ARTICLE{shalaginov2013automatic,
      title={Automatic rule-mining for malware detection employing neuro-fuzzy approach},
      author={Shalaginov, Andrii and Franke, Katrin},
      journal={Norsk informasjonssikkerhetskonferanse (NISK)},
      volume={2013},
      year={2013}
    }

## Dataset

/data has few toy examples and also original features generate during work on the paper. 
The main dataset includes 5 features described in the paper.

## Requirements:

- g++ (tested on v. 4.7.3 and higher)
- STL containers for data operations
- OpenMP for parallel execution (v. 3.1 and higher)
- Doxygen-friendly

## Misc

There is also a possibility to limit a number of execution threads through variable 'maxThreads'

## Experimental CUDA implementation

main.cu - work in progress
