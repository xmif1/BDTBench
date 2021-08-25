//
// Created by Xandru Mifsud on 14/07/2021.
//

#include "root2xgboost.h"

/* Utility function for converting from ROOT's TTree data representation, to xgboost's DMatrix representation.
 * Furthermore,
 * (i)  We also extract the number of signal and background events.
 * (ii) We calculate balanced weights between the signal and background datasets.
 *
 * All this data is wrapped in an xgboost_data instance, the internals of which can be used with xgboost's C-api.
 *
 * The passed vector<string> of variable names specifies which branches to extract from the respective TTree instances.
 */
xgboost_data* ROOTToXGBoost(TTree& signal_tree, TTree& background_tree, vector<string>& variables, const Float_t* sig_weight, const Float_t* bgd_weight){

    // Represent signal TTree as a data frame
    ROOT::RDataFrame sig_dframe(signal_tree);
    auto n_sig = sig_dframe.Count(); // and extract the number of signal events

    // Represent background TTree as a data frame
    ROOT::RDataFrame bgd_dframe(background_tree);
    auto n_bgd = bgd_dframe.Count(); // and extract the number of background events

    const auto n_vars = variables.size(); // count the number of vars

    // Initialise Float_t array to hold a 2-dim representation of the signal and background trees
    Float_t sb_mat[*n_sig + *n_bgd][n_vars];

    // Loop across the variables and events, populating sb_mat resulting in a 2-dim representation of the signal and
    // background trees
    Long64_t i; Long64_t j = 0;
    for(auto& var: variables){
        i = 0;
        for(auto& sig_mat_ij: sig_dframe.Take<Float_t>(var)){ // first n_sig rows will be signal data
            sb_mat[i][j] = sig_mat_ij;
            i++;
        }
        for(auto& bgd_mat_ij: bgd_dframe.Take<Float_t>(var)){ // and the following n_bgd rows will be background data
            sb_mat[i][j] = bgd_mat_ij;
            i++;
        }

        j++;
    }

    auto data = new xgboost_data(*n_sig, *n_bgd); // maintains xgboost readable data

    Float_t sw; // unless given a weight for signal data, we calculate a balanced weight
    if(sig_weight == nullptr){
        sw = 1.0 + (*n_bgd/(*n_sig));
    }else{
        sw = *sig_weight;
    }

    // Set the weight for the signal data, and the labels to 0.0
    for(Long64_t k = 0; k < *n_sig; k++){ (data->labels)[k] = 0.0; (data->weights)[k] = sw; }

    Float_t bw; // unless given a weight for background data, we calculate a balanced weight
    if(bgd_weight == nullptr){
        bw = 1.0 + (*n_sig/(*n_bgd));
    }else{
        bw = *bgd_weight;
    }

    // Set the weight for the background data, and the labels to 1.0
    for(Long64_t k = *n_sig; k < (*n_sig + *n_bgd); k++){ (data->labels)[k] = 1.0; (data->weights)[k] = bw; }

    // Populate the DMatrix datastructure held in the xgboost_data instance...
    safe_xgboost(XGDMatrixCreateFromMat((Float_t*) sb_mat, *n_sig + *n_bgd, n_vars, 0, &((data->sb_dmats)[0])))
    safe_xgboost(XGDMatrixSetFloatInfo((data->sb_dmats)[0], "label", data->labels, *n_sig + *n_bgd))

    return data;
}

/* Utility function for converting from ROOT's DataSetInfo instance, to xgboost's DMatrix representation.
 * Furthermore,
 * (i)   We also extract the number of signal and background events.
 * (ii)  We calculate balanced weights between the signal and background datasets.
 * (iii) We are capable of carrying out additional filtering on the data by filtering out events whose type does not
 *       match the tree type specified, in particular either kTesting or kTraining are supported presently.
 *
 * All this data is wrapped in an xgboost_data instance, the internals of which can be used with xgboost's C-api.
 *
 * The passed vector<string> of variable names specifies which branches to extract from the respective TTree instances.
 */
xgboost_data* ROOTToXGBoost(const TMVA::DataSetInfo& dataset_info, TMVA::Types::ETreeType type){
    TMVA::DataSet* dataset = dataset_info.GetDataSet(); // reference to DataSet instance being filtered

    const auto n_vars = dataset->GetNVariables(); // get number of variables
    Long64_t n_sig, n_bgd; // maintain the number of signal and background events respectively

    // Depending on whether we are keeping testing or training events, set the number of such signal and background events
    if(type == TMVA::Types::kTesting){
        n_sig = dataset->GetNEvtSigTest();
        n_bgd = dataset->GetNEvtBkgdTest();
    }else if(type == TMVA::Types::kTraining){
        n_sig = dataset->GetNEvtSigTrain();
        n_bgd = dataset->GetNEvtBkgdTrain();
    }else{
        throw runtime_error("Unexpected treeType (must be either kTesting or kTraining).");
    }

    // Initialise Float_t array to hold a 2-dim representation of the signal and background trees
    Float_t sb_mat[n_sig + n_bgd][n_vars];

    auto data = new xgboost_data(n_sig, n_bgd); // maintains xgboost readable data

    Long64_t i = 0;
    for(auto& event: dataset->GetEventCollection(type)){ // Notice here that unlike the TTree variant of the function,
                                                         // the rows will be mixed signal and background events
        // Populate the 2d matrix...
        for(Long64_t j = 0; j < n_vars; j++){
            sb_mat[i][j] = event->GetValue(j);
        }


        if(dataset_info.IsSignal(event)){ // Set the weight for the signal data, and the labels to 0.0
            (data->labels)[i] = 0.0;
            (data->weights)[i] = event->GetOriginalWeight();
        }else{
            (data->labels)[i] = 1.0; // Set the weight for the background data, and the labels to 1.0
            (data->weights)[i] = event->GetOriginalWeight();
        }

        i++;
    }

    // Populate the DMatrix datastructure held in the xgboost_data instance...
    safe_xgboost(XGDMatrixCreateFromMat((Float_t*) sb_mat, n_sig + n_bgd, n_vars, 0, &((data->sb_dmats)[0])))
    safe_xgboost(XGDMatrixSetFloatInfo((data->sb_dmats)[0], "label", data->labels, n_sig + n_bgd))

    return data;
}

// Simple trainer using the XGBoost C-api, which returns a trained BoosterHandle instance.
BoosterHandle xgboost_train(xgboost_data* data, xgbooster_opts* opts, UInt_t n_iter){
    BoosterHandle booster;
    safe_xgboost(XGBoosterCreate(data->sb_dmats, 1, &booster)) // Initialise a BoosterHandle instance

    for(auto opt: *opts){ // For each specified option, set the booster parameters...
        safe_xgboost(XGBoosterSetParam(booster, opt.first, opt.second))
    }

    // Then train the booster for the specified number of iterations
    for(UInt_t iter = 0; iter < n_iter; iter++){
        safe_xgboost(XGBoosterUpdateOneIter(booster, iter, (data->sb_dmats)[0]))
    }

    return booster;
}