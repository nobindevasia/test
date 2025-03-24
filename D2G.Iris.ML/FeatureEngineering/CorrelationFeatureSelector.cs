using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;

using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.FeatureEngineering;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class CorrelationFeatureSelector : BaseFeatureSelector
    {
        public CorrelationFeatureSelector(MLContext mlContext) : base(mlContext)
        {
        }

        public override Task<(float[][] features, string[] featureNames, string report)> SelectFeatures(
            List<Dictionary<string, object>> data,
            string[] candidateFeatures,     
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            _report.Clear();
            _report.AppendLine("\nCorrelation-based Feature Selection Results:");
            _report.AppendLine("----------------------------------------------");

            try
            {
                var filteredCandidateFeatures = FilterTargetField(candidateFeatures, targetField);
                var targetValues = ExtractTargetValues(data, targetField);
                var featureValuesList = ExtractFeatureValues(data, filteredCandidateFeatures);

                var targetCorrelations = new Dictionary<string, double>();
                for (int i = 0; i < filteredCandidateFeatures.Length; i++)
                {
                    var correlation = Correlation.Pearson(featureValuesList[i], targetValues.ToArray());
                    targetCorrelations[filteredCandidateFeatures[i]] = Math.Abs(correlation);
                }

                var correlationMatrix = Matrix<double>.Build.Dense(
                    filteredCandidateFeatures.Length,
                    filteredCandidateFeatures.Length
                );

                for (int i = 0; i < filteredCandidateFeatures.Length; i++)
                {
                    for (int j = 0; j < filteredCandidateFeatures.Length; j++)
                    {
                        correlationMatrix[i, j] = Correlation.Pearson(
                            featureValuesList[i],
                            featureValuesList[j]
                        );
                    }
                }

                var selectedFeatures = new List<string>();
                var sortedFeatures = targetCorrelations
                    .OrderByDescending(x => x.Value)
                    .Select(x => x.Key)
                    .ToList();

                _report.AppendLine("\nFeatures Ranked by Target Correlation:");
                foreach (var feature in sortedFeatures)
                {
                    _report.AppendLine($"{feature,-40} | {targetCorrelations[feature]:F4}");
                }

                
                foreach (var feature in sortedFeatures)
                {
                    if (selectedFeatures.Count >= config.MaxFeatures)
                        break;

                    bool isHighlyCorrelated = false;
                    foreach (var selectedFeature in selectedFeatures)
                    {
                        var i1 = Array.IndexOf(filteredCandidateFeatures, feature);
                        var i2 = Array.IndexOf(filteredCandidateFeatures, selectedFeature);
                        if (Math.Abs(correlationMatrix[i1, i2]) > config.MulticollinearityThreshold)
                        {
                            isHighlyCorrelated = true;
                            break;
                        }
                    }

                    if (!isHighlyCorrelated)
                    {
                        selectedFeatures.Add(feature);
                    }
                }

                _report.AppendLine($"\nSelection Summary:");
                _report.AppendLine($"Original features: {filteredCandidateFeatures.Length}");
                _report.AppendLine($"Selected features: {selectedFeatures.Count}");
                _report.AppendLine($"Multicollinearity threshold: {config.MulticollinearityThreshold}");

                var features = ConvertToFeatureMatrix(data, selectedFeatures.ToArray());
                return Task.FromResult((features, selectedFeatures.ToArray(), _report.ToString()));
            }
            catch (Exception ex)
            {
                _report.AppendLine($"Error during correlation analysis: {ex.Message}");
                return Task.FromResult((new float[0][], Array.Empty<string>(), _report.ToString()));
            }
        }
    }
}