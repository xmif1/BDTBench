if(ROOT_tmva_FOUND)
   RB_ADD_GBENCHMARK(BoostedDTBenchmarks
      BoostedDTBenchmarks.cxx
      LABEL short
      LIBRARIES Core Tree TreePlayer MathCore RIO XMLIO ROOTDataFrame TMVA XGBoost::XGBoost)
endif()