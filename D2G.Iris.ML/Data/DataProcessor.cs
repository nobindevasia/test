using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.DataBalancing;
using D2G.Iris.ML.FeatureEngineering;

namespace D2G.Iris.ML.Data
{
    public class DataProcessor : IDataProcessor
    {
        public async Task<ProcessedData> ProcessData(
            MLContext mlContext,
            List<Dictionary<string, object>> rawData,
            string[] enabledFields,
            ModelConfig config,
            ISqlHandler sqlHandler)
        {
            Console.WriteLine("\n=============== Processing Data ==============");
 
            bool balancingFirst = config.DataBalancing.ExecutionOrder <= config.FeatureEngineering.ExecutionOrder;

           
            if (config.DataBalancing.Method != DataBalanceMethod.None &&
                config.FeatureEngineering.Method != FeatureSelectionMethod.None)
            {
                Console.WriteLine($"Processing order: {(balancingFirst ?
                    "Data Balancing then Feature Selection" :
                    "Feature Selection then Data Balancing")}");
            }

            var transformedData = rawData;
            string[] finalFeatureNames = enabledFields;
            float[][] selectedFeatures = null;
            string selectionReport = "";

            if (balancingFirst)
            {
                if (config.DataBalancing.Method != DataBalanceMethod.None)
                {
                    Console.WriteLine($"Applying {config.DataBalancing.Method} balancing...");
                    var dataBalancerFactory = new DataBalancerFactory();
                    var dataBalancer = dataBalancerFactory.CreateBalancer(config.DataBalancing.Method);

                    transformedData = await dataBalancer.BalanceDataset(
                        rawData,
                        enabledFields,
                        config.DataBalancing,
                        config.TargetField
                    );

                    Console.WriteLine($"Data balanced. New dataset size: {transformedData.Count} samples");
                }

                // Step 2: Feature Selection (if enabled)
                if (config.FeatureEngineering.Method != FeatureSelectionMethod.None)
                {
                    var featureSelectorFactory = new FeatureSelectorFactory(mlContext);
                    var featureSelector = featureSelectorFactory.CreateSelector(config.FeatureEngineering.Method);

                    var result = await featureSelector.SelectFeatures(
                        transformedData,
                        enabledFields,
                        config.ModelType,
                        config.TargetField,
                        config.FeatureEngineering
                    );

                    selectedFeatures = result.features;
                    finalFeatureNames = result.featureNames;
                    selectionReport = result.report;
                    Console.WriteLine(selectionReport);
                }
                else
                {
                    // If feature selection is disabled, create a basic feature matrix
                    selectedFeatures = transformedData.Select(row =>
                        enabledFields.Select(field =>
                            row.ContainsKey(field) ? Convert.ToSingle(row[field]) : 0.0f).ToArray()
                    ).ToArray();
                    finalFeatureNames = enabledFields;
                    selectionReport = "Feature selection was disabled.";
                }
            }
            else
            {
                // Step 1: Feature Selection (if enabled)
                if (config.FeatureEngineering.Method != FeatureSelectionMethod.None)
                {
                    var featureSelectorFactory = new FeatureSelectorFactory(mlContext);
                    var featureSelector = featureSelectorFactory.CreateSelector(config.FeatureEngineering.Method);

                    var result = await featureSelector.SelectFeatures(
                        rawData,
                        enabledFields,
                        config.ModelType,
                        config.TargetField,
                        config.FeatureEngineering
                    );

                    // Create intermediate dataset with only selected features
                    var featureSelectedData = new List<Dictionary<string, object>>();
                    for (int i = 0; i < rawData.Count; i++)
                    {
                        var newRow = new Dictionary<string, object>();
                        for (int j = 0; j < result.featureNames.Length; j++)
                        {
                            var featName = result.featureNames[j];
                            newRow[featName] = rawData[i].ContainsKey(featName) ?
                                rawData[i][featName] : 0.0f;
                        }
                        newRow[config.TargetField] = rawData[i][config.TargetField];
                        featureSelectedData.Add(newRow);
                    }

                    finalFeatureNames = result.featureNames;
                    selectionReport = result.report;
                    Console.WriteLine(selectionReport);

                    transformedData = featureSelectedData;
                }

                // Step 2: Data Balancing (if enabled)
                if (config.DataBalancing.Method != DataBalanceMethod.None)
                {
                    Console.WriteLine($"Applying {config.DataBalancing.Method} balancing...");
                    var dataBalancerFactory = new DataBalancerFactory();
                    var dataBalancer = dataBalancerFactory.CreateBalancer(config.DataBalancing.Method);

                    transformedData = await dataBalancer.BalanceDataset(
                        transformedData,
                        finalFeatureNames,
                        config.DataBalancing,
                        config.TargetField
                    );

                    Console.WriteLine($"Data balanced. New dataset size: {transformedData.Count} samples");
                }

                // Create final feature matrix
                selectedFeatures = transformedData.Select(row =>
                    finalFeatureNames.Select(field =>
                        row.ContainsKey(field) ? Convert.ToSingle(row[field]) : 0.0f).ToArray()
                ).ToArray();
            }

            // Create final dataset with selected features
            var finalData = new List<Dictionary<string, object>>();
            for (int i = 0; i < selectedFeatures.Length; i++)
            {
                var newRow = new Dictionary<string, object>();
                for (int j = 0; j < finalFeatureNames.Length; j++)
                {
                    newRow[finalFeatureNames[j]] = selectedFeatures[i][j];
                }
                newRow[config.TargetField] = transformedData[i][config.TargetField];
                finalData.Add(newRow);
            }

            if (!string.IsNullOrEmpty(config.Database.OutputTableName))
            {
                try
                {
                    sqlHandler.SaveProcessedDataToTable(
                        config.Database.OutputTableName,
                        finalData,
                        finalFeatureNames,
                        config.TargetField,
                        config.ModelType);
                    Console.WriteLine($"Processed data saved to table: {config.Database.OutputTableName}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error saving processed data: {ex.Message}");
                }
            }

            return new ProcessedData
            {
                Data = finalData,
                FeatureNames = finalFeatureNames,
                OriginalSampleCount = rawData.Count,
                BalancedSampleCount = transformedData.Count,
                FeatureSelectionReport = selectionReport,
                FeatureSelectionMethod = config.FeatureEngineering.Method,
                DataBalancingMethod = config.DataBalancing.Method,
                Features = selectedFeatures,
                DataBalancingExecutionOrder = config.DataBalancing.ExecutionOrder,
                FeatureSelectionExecutionOrder = config.FeatureEngineering.ExecutionOrder
            };
        }
    }
}