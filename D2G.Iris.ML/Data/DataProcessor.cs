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
            var transformedData = new List<Dictionary<string, object>>();

            var dataBalancerFactory = new DataBalancerFactory();
            var dataBalancer = dataBalancerFactory.CreateBalancer(config.DataBalancing.Method);
    
            transformedData = await dataBalancer.BalanceDataset(
                rawData,
                enabledFields,
                config.DataBalancing,
                config.TargetField
            );

            if (config.DataBalancing.Method != DataBalanceMethod.None)
            {
                Console.WriteLine($"Data balanced. New dataset size: {transformedData.Count} samples");
            }


            var featureSelectorFactory = new FeatureSelectorFactory(mlContext);
            var featureSelector = featureSelectorFactory.CreateSelector(config.FeatureEngineering.Method);

            var (selectedFeatures, featureNames, selectionReport) = await featureSelector.SelectFeatures(
                transformedData,
                enabledFields,
                config.ModelType,
                config.TargetField,
                config.FeatureEngineering
            );

            Console.WriteLine(selectionReport);

            
            var finalData = new List<Dictionary<string, object>>();
            for (int i = 0; i < selectedFeatures.Length; i++)
            {
                var newRow = new Dictionary<string, object>();
                for (int j = 0; j < featureNames.Length; j++)
                {
                    newRow[featureNames[j]] = selectedFeatures[i][j];
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
                        featureNames,
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
                FeatureNames = featureNames,
                OriginalSampleCount = rawData.Count,
                BalancedSampleCount = transformedData.Count,
                FeatureSelectionReport = selectionReport,
                FeatureSelectionMethod = config.FeatureEngineering.Method,
                DataBalancingMethod = config.DataBalancing.Method,
                Features = selectedFeatures
            };
        }
    }
}