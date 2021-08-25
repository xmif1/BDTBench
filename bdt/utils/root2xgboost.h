//
// Created by Xandru Mifsud on 14/07/2021.
//

#ifndef ROOT2XGBOOST_ROOT2XGBOOST_H
#define ROOT2XGBOOST_ROOT2XGBOOST_H

#include <xgboost/c_api.h>
#include <ROOT/RDataFrame.hxx>
#include <TMVA/DataLoader.h>
#include <TMVA/DataSetInfo.h>
#include <TMVA/DataSet.h>
#include <TMVA/Factory.h>
#include <TMVA/Types.h>

using namespace std;

/* Simple macro to carry out error checking and handling around xgboost C-api calls
 * See also here: https://github.com/dmlc/xgboost/blob/master/demo/c-api/c-api-demo.c
 */
#define safe_xgboost(call){                                                                      \
    int err = (call);                                                                            \
    if (err != 0){                                                                               \
        throw runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +             \
                            ": error in " + #call + ":" + XGBGetLastError());                    \
    }                                                                                            \
}

/* Struct-type which maintains the necessary data required for, in particular, training and making
 * predictions using xgboost's C-api.
 *
 * free() should be called to 'destruct' an xgboost_data instance; especially important since it
 * frees memory associated with DMatrix instances, which otherwise could lead to memory leaks.
 */
typedef struct xgboost_data{
    // meta-data
    DMatrixHandle sb_dmats[1];
    Float_t* weights;
    Float_t* labels;
    Long64_t n_sig, n_bgd;

    xgboost_data(Long64_t n_sig, Long64_t n_bgd){
        this->n_sig = n_sig;
        this->n_bgd = n_bgd;

        this->weights = new Float_t[n_sig + n_bgd];
        this->labels = new Float_t[n_sig + n_bgd];
    }

    // call for memory management
    void free() const{
        safe_xgboost(XGDMatrixFree(sb_dmats[0]))
    }
} xgboost_data;

typedef pair<char const*, char const*> kv_pair;
typedef vector<kv_pair> xgbooster_opts;

xgboost_data* ROOTToXGBoost(TTree& signal_tree, TTree& background_tree, vector<string>& variables,
                            const Float_t* sig_weight = nullptr, const Float_t* bgd_weight = nullptr);
xgboost_data* ROOTToXGBoost(const TMVA::DataSetInfo& dataset_info, TMVA::Types::ETreeType type);
BoosterHandle xgboost_train(xgboost_data* data, xgbooster_opts* opts, UInt_t n_iter);


#endif //ROOT2XGBOOST_ROOT2XGBOOST_H
