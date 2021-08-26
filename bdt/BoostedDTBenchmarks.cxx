/* Authored by Xandru Mifsud (CERN Summer Student) and Lorenzo Moneta (Summer Project Supervisor) */

#include "TSystem.h"
#include "TTree.h"
#include "TFile.h"

#include "TMVA/RReader.hxx"
#include "TMVA/RTensorUtils.hxx"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/MethodBase.h"
#include "TMVA/Types.h"

#include "benchmark/benchmark.h"

#include "utils/MakeRandomTTree.h"
#include "utils/root2xgboost.h"

using namespace TMVA::Experimental;
using namespace std;

static void BM_TMVA_BDTTraining(benchmark::State &state){
   // Parameters
   UInt_t nVars = 4;
   UInt_t nEvents = 500;
   Bool_t mem_stats = (state.range(0) == 2000) && (state.range(1) == 10) && (state.range(2) == 1);

   // Memory benchmark data placeholder
   ProcInfo_t pinfo;
   Long_t init_mem_res, term_mem_res; init_mem_res = term_mem_res = 0;
   double mem_res = 0.0;

   // Open output file
   TString outfileName( "bdt_tmva_bench_train_output.root" );
   TFile* outputFile = TFile::Open(outfileName, "RECREATE");

   // Set up
   TTree *sigTree = genTree("sigTree", nEvents, nVars,0.3, 0.5, 100);
   TTree *bkgTree = genTree("bkgTree", nEvents, nVars,-0.3, 0.5, 101);

   // Prepare a DataLoader instance, registering the signal and background TTrees
   auto *dataloader = new TMVA::DataLoader("bdt_tmva_bench");
   dataloader->AddSignalTree(sigTree);
   dataloader->AddBackgroundTree(bkgTree);

   // Register variables in dataloader, using naming convention for randomly generated TTrees in MakeRandomTTree.h
   for(UInt_t i = 0; i < nVars; i++){
      string var_name = "var" + to_string(i);
      string var_leaflist = var_name + "/F";

      dataloader->AddVariable(var_name.c_str(), 'D');
   }

   dataloader->PrepareTrainingAndTestTree("",
                  Form("SplitMode=Block:nTrain_Signal=%i:nTrain_Background=%i:!V", nEvents, nEvents));

   // Benchmarking
   UInt_t iter_c = 0;
   for(auto _: state){
      ROOT::EnableImplicitMT(state.range(2));

      // Create factory instance
      auto factory = new TMVA::Factory("bdt_tmva_bench", outputFile,
                                    "Silent:!DrawProgressBar:AnalysisType=Classification");

      // Get current memory usage statistics after setup
      if(mem_stats && iter_c == 0){
         gSystem->GetProcInfo(&pinfo);
         init_mem_res = pinfo.fMemResident;
      }
      
      // Construct training options string
      string opts = "!V:!H:NTrees=" + to_string(state.range(0)) + ":MaxDepth=" + to_string(state.range(1));

      // Train a TMVA method
      string key = to_string(state.range(0)) + "_" + to_string(state.range(1)) + "_" + to_string(state.range(2));
      auto method = factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT_" + key, opts);
      TMVA::Event::SetIsTraining(kTRUE);
      method->TrainMethod();

      // Maintain Memory statistics (independent from Google Benchmark)
      if(mem_stats && iter_c == 0){
         gSystem->GetProcInfo(&pinfo);
         term_mem_res = pinfo.fMemResident;
         mem_res += (double) (term_mem_res - init_mem_res);
      }

      TMVA::Event::SetIsTraining(kFALSE);
      method->Data()->DeleteAllResults(TMVA::Types::kTraining, method->GetAnalysisType());

      // Destroy factory entirely
      factory->DeleteAllMethods();
      factory->fMethodsMap.clear();
      delete factory;

      // DEBUG
      // cout << "[DEBUG] " << key << ": res_mem_init = " << (double) init_mem_res << ", res_mem_term = " << (double) term_mem_res << endl;

      iter_c++;
   }

   if(mem_stats){
      mem_res *= iter_c;
      state.counters["Resident Memory"] = benchmark::Counter(mem_res, benchmark::Counter::kAvgIterations);
   }

   // Teardown
   delete sigTree;
   delete bkgTree;

   outputFile->Close();
   delete outputFile;
}
//BENCHMARK(BM_TMVA_BDTTraining)->ArgsProduct({{2000, 1000, 400, 100}, {10, 8, 6, 4, 2}, {1, 4, 8, 16}});
BENCHMARK(BM_TMVA_BDTTraining)->ArgsProduct({{2000, 1000, 400, 100}, {10, 8, 6, 4, 2}, {1}});

static void BM_XGBOOST_BDTTraining(benchmark::State &state){
   // Parameters
   UInt_t nVars = 4;
   UInt_t nEvents = 500;
   Bool_t mem_stats = (state.range(0) == 2000) && (state.range(1) == 10);

   // Memory benchmark data placeholder
   ProcInfo_t pinfo;
   Long_t init_mem_res, term_mem_res; init_mem_res = term_mem_res = 0;
   double mem_res = 0.0;

   // Open output file
   TString outfileName( "bdt_xgb_bench_train_output.root" );
   TFile* outputFile = TFile::Open(outfileName, "RECREATE");

   // Set up
   TTree *sigTree = genTree("sigTree", nEvents, nVars,0.3, 0.5, 100);
   TTree *bkgTree = genTree("bkgTree", nEvents, nVars,-0.3, 0.5, 101);

   // Prepare a DataLoader instance, registering the signal and background TTrees
   auto *dataloader = new TMVA::DataLoader("bdt_xgb_bench");
   dataloader->AddSignalTree(sigTree);
   dataloader->AddBackgroundTree(bkgTree);

   // Register variables in dataloader, using naming convention for randomly generated TTrees in MakeRandomTTree.h
   for(UInt_t i = 0; i < nVars; i++){
      string var_name = "var" + to_string(i);
      string var_leaflist = var_name + "/F";

      dataloader->AddVariable(var_name.c_str(), 'D');
   }

   dataloader->PrepareTrainingAndTestTree("",
                  Form("SplitMode=Block:nTrain_Signal=%i:nTrain_Background=%i:!V", nEvents, nEvents));

   // Extract the training data set and convert it to an XGBoost readable format
   xgboost_data* xg_train_data = ROOTToXGBoost(dataloader->GetDefaultDataSetInfo(), TMVA::Types::kTraining);

   // Benchmarking
   UInt_t iter_c = 0;
   for(auto _: state){
      // Set the options for the BoosterHandle instance that will be trained...
      xgbooster_opts opts;
      opts.push_back(kv_pair("max_depth", std::to_string((int) state.range(1)).c_str()));

      // Get current memory usage statistics after setup
      if(mem_stats && iter_c == 0){
         gSystem->GetProcInfo(&pinfo);
         init_mem_res = pinfo.fMemResident;
      }

      // Call the simple trainer function we implemented based on XGBoost's C-api...
      BoosterHandle xgbooster = xgboost_train(xg_train_data, &opts, state.range(0));

      // Maintain Memory statistics (independent from Google Benchmark)
      if(mem_stats && iter_c == 0){
         gSystem->GetProcInfo(&pinfo);
         term_mem_res = pinfo.fMemResident;
         mem_res += (double) (term_mem_res - init_mem_res);
      }

      iter_c++;

      // Save XGBoost trained booster instance
      string fname = "BDT_" + to_string(state.range(0)) + "_" + to_string(state.range(1)) + ".model";
      safe_xgboost(XGBoosterSaveModel(xgbooster, fname.c_str()))

      // Free XGBoost related memory
      safe_xgboost(XGBoosterFree(xgbooster))
   }

   if(mem_stats){
      mem_res *= iter_c;
      state.counters["Resident Memory"] = benchmark::Counter(mem_res, benchmark::Counter::kAvgIterations);
   }

   // Teardown
   delete sigTree;
   delete bkgTree;
   xg_train_data->free();

   outputFile->Close();
   delete outputFile;
}
BENCHMARK(BM_XGBOOST_BDTTraining)->ArgsProduct({{2000, 1000, 400, 100}, {10, 8, 6, 4, 2}});

static void BM_TMVA_BDTTesting(benchmark::State &state){
   // Parameters
   UInt_t nVars = 4;
   UInt_t nEvents = 500;
   Bool_t mem_stats = (state.range(0) == 2000) && (state.range(1) == 10) && (state.range(2) == 1);

   // Memory benchmark data placeholder
   ProcInfo_t pinfo;
   Long_t init_mem_res, term_mem_res; init_mem_res = term_mem_res = 0;
   double mem_res = 0.0;

   // Open output file
   TString outfileName( "bdt_tmva_bench_test_output.root" );
   TFile* outputFile = TFile::Open(outfileName, "RECREATE");

   // Set up
   auto inputFile = new TFile("bdt_tmva_bench_test_input.root","RECREATE");
   TTree *testTree = genTree("testTree", nEvents, nVars,0.3, 0.5, 102, false);
   testTree->Write();
   delete testTree;
   inputFile->Close();
   delete inputFile;

   ROOT::RDataFrame testDF("testTree","bdt_tmva_bench_test_input.root");
   auto testTensor = AsTensor<Float_t>(testDF);

   // Benchmarking
   UInt_t iter_c = 0;
   for(auto _: state){
      ROOT::EnableImplicitMT(state.range(2));

      // Test a TMVA method via RReader
      string key = to_string(state.range(0)) + "_" + to_string(state.range(1)) + "_" + to_string(state.range(2));

      // Get current memory usage statistics after setup
      if(mem_stats && iter_c == 0){
         gSystem->GetProcInfo(&pinfo);
         init_mem_res = pinfo.fMemResident;
      }

      RReader model("./bdt_tmva_bench/weights/bdt_tmva_bench_BDT_" + key + ".weights.xml");
      model.Compute(testTensor);

      // Maintain Memory statistics (independent from Google Benchmark)
      if(mem_stats && iter_c == 0){
         gSystem->GetProcInfo(&pinfo);
         term_mem_res = pinfo.fMemResident;
         mem_res += (double) (term_mem_res - init_mem_res);
      }

      iter_c++;
   }

   if(mem_stats){
      mem_res *= iter_c;
      state.counters["Resident Memory"] = benchmark::Counter(mem_res, benchmark::Counter::kAvgIterations);
   }

   // Teardown
   outputFile->Close();
}
//BENCHMARK(BM_TMVA_BDTTesting)->ArgsProduct({{2000, 1000, 400, 100}, {10, 8, 6, 4, 2}, {1, 4, 8, 16}});
BENCHMARK(BM_TMVA_BDTTesting)->ArgsProduct({{2000, 1000, 400, 100}, {10, 8, 6, 4, 2}, {1}});

static void BM_XGBOOST_BDTTesting(benchmark::State &state){
   // Parameters
   UInt_t nVars = 4;
   UInt_t nEvents = 250; // half size since DataLoader requires test data to be split between signal and background...
   Bool_t mem_stats = (state.range(0) == 2000) && (state.range(1) == 10);

   // Memory benchmark data placeholder
   ProcInfo_t pinfo;
   Long_t init_mem_res, term_mem_res; init_mem_res = term_mem_res = 0;
   double mem_res = 0.0;

   // Open output file
   TString outfileName( "bdt_xgb_bench_test_output.root" );
   TFile* outputFile = TFile::Open(outfileName, "RECREATE");

   // Set up (create one additional event to silence TMVA DataLoader error for no training events)
   TTree *testTree = genTree("testTree", nEvents + 1, nVars, 0.3, 0.5, 102);
   TTree *trainBKGTree = genTree("bkgTree", nEvents + 1, nVars, 0.3, 0.5, 103);

   // Prepare a DataLoader instance, registering the testing TTrees
   auto *dataloader = new TMVA::DataLoader("bdt_xgb_bench");
   dataloader->AddSignalTree(testTree);
   dataloader->AddBackgroundTree(trainBKGTree);

   // Register variables in dataloader, using naming convention for randomly generated TTrees in MakeRandomTTree.h
   for(UInt_t i = 0; i < nVars; i++){
      string var_name = "var" + to_string(i);
      string var_leaflist = var_name + "/F";

      dataloader->AddVariable(var_name.c_str(), 'D');
   }

   // Prepare the testing data set and convert it to an XGBoost readable format
   dataloader->PrepareTrainingAndTestTree("",
      Form("SplitMode=Block:nTrain_Signal=%i:nTrain_Background=%i:nTest_Signal=%i:nTest_Background=%i:!V", 1, 1, nEvents, nEvents));
   xgboost_data* xg_test_data = ROOTToXGBoost(dataloader->GetDefaultDataSetInfo(), TMVA::Types::kTesting);

   // Benchmarking
   UInt_t iter_c = 0;
   for(auto _: state){
      // Load the trained booster model...
      string fname = "BDT_" + to_string(state.range(0)) + "_" + to_string(state.range(1)) + ".model";
      BoosterHandle xgbooster;
      XGBoosterCreate(0, 0, &xgbooster);
      safe_xgboost(XGBoosterLoadModel(xgbooster, fname.c_str()))

      // Get current memory usage statistics after setup
      if(mem_stats && iter_c == 0){
         gSystem->GetProcInfo(&pinfo);
         init_mem_res = pinfo.fMemResident;
      }

      // Prepare the necessary data structures and carry out the predictions on the (converted) testing data set...
      bst_ulong output_length;
      const Float_t *output_result;
      safe_xgboost(XGBoosterPredict(xgbooster, (xg_test_data->sb_dmats)[0], 0, 0, &output_length, &output_result))

      // Maintain Memory statistics (independent from Google Benchmark)
      if(mem_stats && iter_c == 0){
         gSystem->GetProcInfo(&pinfo);
         term_mem_res = pinfo.fMemResident;
         mem_res += (double) (term_mem_res - init_mem_res);
      }

      iter_c++;

      // Lastly, free any XGBoost related data structures to prevent memory leaks.
      xg_test_data->free();
      safe_xgboost(XGBoosterFree(xgbooster));
   }

   if(mem_stats){
      mem_res *= iter_c;
      state.counters["Resident Memory"] = benchmark::Counter(mem_res, benchmark::Counter::kAvgIterations);
   }

   // Teardown
   delete testTree;
   delete trainBKGTree;

   outputFile->Close();
   delete outputFile;
}
BENCHMARK(BM_XGBOOST_BDTTesting)->ArgsProduct({{2000, 1000, 400, 100}, {10, 8, 6, 4, 2}});

BENCHMARK_MAIN();
