/* 
 * File:   main.cu
 * \author Andrey Shalaginov <andrii.shalaginov@gmail.com>
 * \brief: Implementation of the fuzzy rules construction module using neuro-fuzzy approach using CUDA interaction
 * Created on April 10, 2013, 10:01 AM
 */

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdexcept>
#include <omp.h>//OpenMP
/**
 * Include STL
 */
#include <string>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>    // std::find
/**
 * Include CUDA
 */
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/inner_product.h>
//#include <thrust/reduce.h>
#include <cuda.h>

#define N  595
#define dimTotal  36
#define dim 10
#define nFuzzySets 5


using namespace std;

//Global variable (ANN output) for using on GPU
__device__ float output;

/**
 * Sigmoid activation function
 * @param x linear combiner value
 * @return activation function values
 */
float sigmoidFunction(float x) {
    return 1 / (1 + exp(-x));
}

/**
 * Supplementary function for calculation oposit class label
 * @param classID ID of the class: 0 or 1
 * @return opposite class ID
 */
int oppositClass(short int classID) {
    if (classID == 0)
        return 1;
    else
        return 0;
}

/**
 * Fuzzification function using 3 or 5 linguistic terms
 * @param x value to be fuzzified
 * @param stDev standard deviation of the dataset
 * @param mean mean of the dataset
 * @param idSet ID of the linguistic term
 * @return membership degree of the value x
 */
__host__ __device__ float fuzzificationFunction(float x, float stDev, float mean, char idSet) {
    float tmp1;
    if (nFuzzySets == 2) {
        if (idSet == 0)
            tmp1 = pow((x - mean - stDev) / (stDev), 2);
        else if (idSet == 1)
            tmp1 = pow((x - mean + stDev) / (stDev), 2);
    } else if (nFuzzySets == 3) {
        if (idSet == 0)
            tmp1 = pow((x - mean - stDev) / (stDev), 2);
        else if (idSet == 1)
            tmp1 = pow((x - mean) / (stDev), 2);
        else if (idSet == 2)
            tmp1 = pow((x - mean + stDev) / (stDev), 2);

    } else if (nFuzzySets == 5) {
        if (idSet == 0)
            tmp1 = pow((x - mean - 2 * stDev) / (stDev), 2);
        else if (idSet == 1)
            tmp1 = pow((x - mean - stDev) / (stDev), 2);
        else if (idSet == 2)
            tmp1 = pow((x - mean) / (stDev), 2);
        else if (idSet == 3)
            tmp1 = pow((x - mean + stDev) / (stDev), 2);
        else if (idSet == 4)
            tmp1 = pow((x - mean + 2 * stDev) / (stDev), 2);
    }

    return 1 / exp(tmp1);
}

/*
 * 
 * Adjustment of weights for constructed rules,  kernel implementation for CUDA
 */
__global__ void weightsAdjustment(float *weights, float *classID, float *rules, int numberRules, int m) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < numberRules)
        weights[k] += 0.1 * (classID[m] - output) * rules[k];
    else
        output = 0;
}

/*
 * Linear combiner function, kernel implementation for CUDA
 */
__global__ void activationFunction(float *weights, float *input, float *means, float *stDev, float *rules, int numberRules, int m) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < numberRules) {
        float tmp1 = 1;
        tmp1 = fuzzificationFunction(input[m * dim ], means[0], stDev[0], (int) k / (int) pow((double) nFuzzySets, (double) dim - 1));
        for (int l = 1; l < dim; l++)
            tmp1 *= fuzzificationFunction(input[m * dim + l], means[l], stDev[l], (int) k / (int) pow((double) nFuzzySets, (double) dim - l - 1) % (int) pow((double) nFuzzySets, (double) 1));

        rules[k] = tmp1;
        output += weights[k] * rules[k];
    }
}

/**
 * SupplementaryFunction for determining the linguistic term name based on its ID
 * @param id ID of the linguistic term
 * @return name of linguistic term
 */
string idToLingustic(short int id) {
    string strTmp = "";
    if (nFuzzySets == 2) {
        if (id == 0)
            strTmp = "   LOW";
        else if (id == 1)
            strTmp = "HIGH";

    } else
        if (nFuzzySets == 3) {
        if (id == 0)
            strTmp = "   LOW";
        else if (id == 1)
            strTmp = "MEDIUM";
        else if (id == 2)
            strTmp = "  HIGH";
    } else if (nFuzzySets == 5) {
        if (id == 0)
            strTmp = " VERY LOW";
        else if (id == 1)
            strTmp = "      LOW";
        else if (id == 2)
            strTmp = "   MEDIUM";
        else if (id == 3)
            strTmp = "     HIGH";
        else if (id == 4)
            strTmp = "VERY HIGH";
    }
    return strTmp;
}

/**
 * Main function
 * @param argc
 * @param argv
 * @return 
 */
int main(int argc, char** argv) {
    vector< vector<float> > input, inputTest; //train and test datasets patterns
    vector<float> tmp; //temporary variable
    vector<float> classID, classIDTest; //train and test classes
    vector<float> means, means1, means2; //means for all dataset and both classes
    vector<float> stDev, stDev1, stDev2; //StDev for all dataset and both classes
    vector<float> weights; //weights of each rule
    double tmp1, tmp2; //temporary variables
    long long int numberRules = (long long int) pow(nFuzzySets, dim); //total amount of extracted fuzzy rules
    short int cl; //temporary variable
    vector<float> rules(numberRules); //extracted fuzzy rules
    FILE *pFileTrain, *pFileTest; //pointer to train and test files

    //Read files with training and testing datasets
    if ((pFileTrain = fopen("data/appsFeatures_train.txt", "rt")) == NULL)
        //if ((pFileTrain = fopen("data/train_setosa_versicolor.txt", "rt")) == NULL)
        //if ((pFileTrain = fopen("data/train_setosa_virginica.txt", "rt")) == NULL)
        //if ((pFileTrain = fopen("data/train_versicolor_virginica.txt", "rt")) == NULL)
        puts("Error while opening input train file!");

    if ((pFileTest = fopen("appsFeatures_test.txt", "rt")) == NULL)
        //if ((pFileTest = fopen("data/test_setosa_versicolor.txt", "rt")) == NULL)
        //if ((pFileTest = fopen("data/test_setosa_virginica.txt", "rt")) == NULL)
        //if ((pFileTest = fopen("data/test_versicolor_virginica.txt", "rt")) == NULL)
        puts("Error while opening input test file!");

    //Parsing train file content into data structure
    while (!feof(pFileTrain)) {
        tmp.clear();
        for (int i = 0; i < dimTotal; i++) {
            fscanf(pFileTrain, "%lf ", &tmp1);
            if (i < dim) {

                tmp.push_back((float) tmp1);
            }
        }
        input.push_back(tmp);
        fscanf(pFileTrain, "%hd ", &cl);
        classID.push_back(cl);
    }

    //Parsing test file content into data structure
    while (!feof(pFileTest)) {

        tmp.clear();
        for (int i = 0; i < dimTotal; i++) {
            fscanf(pFileTest, "%lf ", &tmp1);
            if (i < dim) {
                tmp.push_back((float) tmp1);
            }
        }
        inputTest.push_back(tmp);
        fscanf(pFileTest, "%hd ", &cl);
        classIDTest.push_back(cl);
    }
    //Frees & Closes
    fclose(pFileTrain);
    fclose(pFileTest);


    //Calculate mean and standard deviation for both classes separately
    //Means 
    int numEntriesTmp1, numEntriesTmp2;
    for (int j = 0; j < dim; j++) {

        tmp1 = 0;
        tmp2 = 0;
        numEntriesTmp1 = 0;
        numEntriesTmp2 = 0;
        for (int i = 0; i < N; i++) {
            if (classID[i] == 0) {
                tmp1 += input[i][j];
                numEntriesTmp1++;
            } else if (classID[i] == 1) {
                tmp2 += input[i][j];
                numEntriesTmp2++;
            }
        }
        tmp1 = tmp1 / numEntriesTmp1;
        means1.push_back(tmp1);
        tmp2 = tmp2 / numEntriesTmp2;
        means2.push_back(tmp2);
    }

    //StDev  
    for (int j = 0; j < dim; j++) {
        tmp1 = 0;
        tmp2 = 0;
        numEntriesTmp1 = 0;
        numEntriesTmp2 = 0;
        for (int i = 0; i < N; i++) {
            if (classID[i] == 0) {
                tmp1 += (input[i][j] - means1[j])*(input[i][j] - means1[j]);
                numEntriesTmp1++;

            } else if (classID[i] == 1) {
                tmp2 += (input[i][j] - means2[j])*(input[i][j] - means2[j]);
                numEntriesTmp2++;
            }
        }
        tmp1 = sqrt(tmp1 / numEntriesTmp1);
        stDev1.push_back(tmp1);
        tmp2 = sqrt(tmp2 / numEntriesTmp2);
        stDev2.push_back(tmp2);
    }

    //Calculate overall mean and standard deviation
    //Mean
    for (int j = 0; j < dim; j++) {
        tmp1 = 0;
        for (int i = 0; i < N; i++) {
            tmp1 += input[i][j];
        }
        tmp1 = tmp1 / N;
        means.push_back(tmp1);
    }

    //StDev
    for (int j = 0; j < dim; j++) {
        tmp1 = 0;
        for (int i = 0; i < N; i++) {
            tmp1 += (input[i][j] - means[j])*(input[i][j] - means[j]);

        }
        tmp1 = sqrt(tmp1 / (N));
        stDev.push_back(tmp1);
    }

    //Weights Initialization
    for (long long int j = 0; j < numberRules; j++)
        weights.push_back(1 / (numberRules));

    //CUDA required pointers to memory on device and on host
    float *Dinput, *Dmeans, *DstDev, *Dweights, *Drules, *DclassID;
    float HclassID[N], Hweights[numberRules], Hrules[numberRules], Hmeans[dim], HstDev[dim], Hinput[N * dim ];

    //allocate the memory for data arrays on the GPU
    cudaMalloc((void**) &Dinput, N * dim * sizeof (float));
    cudaMalloc((void**) &Dmeans, dim * sizeof (float));
    cudaMalloc((void**) &DstDev, dim * sizeof (float));
    cudaMalloc((void**) &DclassID, N * sizeof (float));

    //copy the arrays from RAM to the GPU memory
    //linearization of two-dimensional array
    for (int y = 0; y < N; y++)
        for (int x = 0; x < dim; x++)
            Hinput[y * dim + x] = input[y][x];
    cudaMemcpy(Dinput, Hinput, N * dim * sizeof (float), cudaMemcpyHostToDevice);

    std::copy(means.begin(), means.end(), Hmeans);
    cudaMemcpy(Dmeans, Hmeans, dim * sizeof (float), cudaMemcpyHostToDevice);

    std::copy(stDev.begin(), stDev.end(), HstDev);
    cudaMemcpy(DstDev, HstDev, dim * sizeof (float), cudaMemcpyHostToDevice);

    std::copy(classID.begin(), classID.end(), HclassID);
    cudaMemcpy(DclassID, HclassID, N * sizeof (float), cudaMemcpyHostToDevice);

    // copy the array 'c' back from the GPU to the CPU
    cudaMalloc((void**) &Dweights, numberRules * sizeof (float));
    std::copy(weights.begin(), weights.end(), Hweights);
    cudaMemcpy(Dweights, Hweights, numberRules * sizeof (float), cudaMemcpyHostToDevice);

    cudaMalloc((void**) &Drules, numberRules * sizeof (float));
    std::copy(rules.begin(), rules.end(), Hrules);
    cudaMemcpy(Drules, Hrules, numberRules * sizeof (float), cudaMemcpyHostToDevice);

    //Neural Network learning
    for (int i = 0; i < 100; i++) {
        for (int m = 0; m < N; m++) {
            //Calculate activation function using 512 GPU threads
            activationFunction << <512 / 256, 256 >> > (Dweights, Dinput, Dmeans, DstDev, Drules, numberRules, m);
            //Adjust rules weights using 512 GPU threads
            weightsAdjustment << <512 / 256, 256 >> >(Dweights, DclassID, Drules, numberRules, m);
        }
    }

    // copy the arrays back from the GPU to the CPU
    cudaMemcpy(Hweights, Dweights, numberRules * sizeof (float), cudaMemcpyDeviceToHost);
    std::copy(Hweights, Hweights + numberRules, weights.begin());
    cudaMemcpy(Hrules, Drules, numberRules * sizeof (float), cudaMemcpyDeviceToHost);
    std::copy(Hrules, Hrules + numberRules, weights.begin());

    //Free the memory allocated on the GPU
    cudaFree(DclassID);
    cudaFree(Dmeans);
    cudaFree(DstDev);
    cudaFree(Dinput);
    cudaFree(Dweights);
    cudaFree(Drules);

    //Sorting constructed rules according to fuzzy-neuro weight value
    vector<short int> rulesAtomId;
    map<float, vector<short int> > rulesMap;
    for (long long int k = 0; k < numberRules; k++) {
        rulesAtomId.clear();
        rulesAtomId.push_back((int) k / (int) pow(nFuzzySets, dim - 1));
        for (int l = 1; l < dim; l++) {
            rulesAtomId.push_back((int) k / (int) pow(nFuzzySets, dim - l - 1) % (int) pow(nFuzzySets, 1));
        }
        rulesMap.insert(std::pair<float, vector<short int> >(weights[k] + 1e-5 * k, rulesAtomId));
    }

    //Assigning Class ID to extracted rules
    std::map<float, vector<short int> >::reverse_iterator it;
    vector< vector<short int> >extractedRules;
    vector<short int>extractedRulesClasses;
    vector<short int> ruleTmp;
    for (it = rulesMap.rbegin(); it != rulesMap.rend(); it++) {
        ruleTmp.clear();
        float degC1 = 1, degC2 = 1;
        //Membership degree calculation for each class
        for (int l = 0; l < dim; l++) {
            degC1 *= fuzzificationFunction(means1[l], stDev[l], means[l], it->second[l]);
            degC2 *= fuzzificationFunction(means2[l], stDev[l], means[l], it->second[l]);
            ruleTmp.push_back(it->second[l]);
        }
        //Defining class
        if (degC1 > degC2) {
            extractedRulesClasses.push_back(0);
        } else {
            extractedRulesClasses.push_back(1);
        }
        extractedRules.push_back(ruleTmp);
    }

    //Erase irrelevant rules
    int u = 0, numberExtracedRulesMainClass;
    short int detectedMainClass = extractedRulesClasses[0];
    printf("Main class: %d \n", detectedMainClass);
    ;
    for (u = 0; u < extractedRulesClasses.size() - 1; u++) {
        if (extractedRulesClasses[u] != extractedRulesClasses[0])
            break;
    }
    extractedRules.erase(extractedRules.begin() + u, extractedRules.end());
    extractedRulesClasses.erase(extractedRulesClasses.begin() + u, extractedRulesClasses.end());

    //Print constructed rules
    numberExtracedRulesMainClass = u;
    u = 0;
    printf("rule weight  | feature1  |  feature2  |  feature3  |  feature4 | Class  (m.degree Cl1  m.degree Cl2)\n");
    for (it = rulesMap.rbegin(); it != rulesMap.rend() && u < numberExtracedRulesMainClass; it++) {
        u++;
        printf("w: %.6f | ", it->first);
        float degC1 = 1, degC2 = 1;
        for (int l = 0; l < dim; l++) {
            if (l > 0)
                printf(" and ");
            printf("%s  ", idToLingustic(it->second[l]).c_str());
            degC1 *= fuzzificationFunction(means1[l], stDev[l], means[l], it->second[l]);
            degC2 *= fuzzificationFunction(means2[l], stDev[l], means[l], it->second[l]);
        }
        //printing class of the rule
        if (degC1 > degC2)
            printf("=> Class0");
        else
            printf("=> Class1");

        printf(" (%f     ", degC1);
        printf(" %f    )\n", degC2);
    }

    printf("Rules main class: %d\n", extractedRulesClasses[0]);
    printf("Amount of constructed rules: %d\n", (int) pow(nFuzzySets, dim));
    printf("Amount of selected rules: %ld\n", extractedRulesClasses.size());


    //Classification validation
    float classificationAccuracy = 0;
    short int actualClass;
    for (int i = 0; i < inputTest.size(); i++) {
        vector<map<float, short int > >linguistic_membership_var;
        map<float, short int > lingusticAttributeSorted;
        ruleTmp.clear();
        //Checking the fuzzy rule
        for (int l = 0; l < dim; l++) {
            lingusticAttributeSorted.clear();
            for (int m = 0; m < nFuzzySets; m++) {
                lingusticAttributeSorted.insert(std::pair<float, short int >(fuzzificationFunction(inputTest[i][l], means[l], stDev[l], m), m));
            }

            ruleTmp.push_back(lingusticAttributeSorted.rbegin()->second);
            linguistic_membership_var.push_back(lingusticAttributeSorted);

        }

        //Searching the input pattern rule among previously constructed
        std::vector<vector <short int> >::iterator iter;
        iter = find(extractedRules.begin(), extractedRules.end(), ruleTmp);
        size_t classIndex = std::distance(extractedRules.begin(), iter);
        if (iter != extractedRules.end() && classIndex != extractedRules.size())
            actualClass = detectedMainClass;
        else
            actualClass = oppositClass(detectedMainClass);

        printf("Predicted class id of the sample: %d (actual: %.0f)\n", actualClass, classIDTest[i]);

        if (classIDTest[i] == actualClass)
            classificationAccuracy++;
    }

    printf("Accuracy: %.2f %% ", classificationAccuracy / inputTest.size() * 100);
    return 0;
}
