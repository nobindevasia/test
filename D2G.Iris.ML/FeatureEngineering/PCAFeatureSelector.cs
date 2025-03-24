using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.FeatureEngineering;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class PCAFeatureSelector : BaseFeatureSelector
    {
        public PCAFeatureSelector(MLContext mlContext) : base(mlContext)
        {
        }

        public override async Task<(float[][] features, string[] featureNames, string report)> SelectFeatures(
            List<Dictionary<string, object>> data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            _report.Clear();
            _report.AppendLine("PCA Transformation Results");
            _report.AppendLine("--------------------------------");

            var filteredCandidateFeatures = FilterTargetField(candidateFeatures, targetField);
            var rows = data.Select(dict => new InputData
            {
                Features = filteredCandidateFeatures.Select(feature =>
                    dict.ContainsKey(feature) && dict[feature] != null
                        ? Convert.ToSingle(dict[feature])
                        : 0.0f)
                    .ToArray()
            }).ToList();

            var schema = SchemaDefinition.Create(typeof(InputData));
            schema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, filteredCandidateFeatures.Length);

            var dataView = _mlContext.Data.LoadFromEnumerable(rows, schema);

            var pipeline = _mlContext.Transforms.NormalizeMeanVariance("Features")
                .Append(_mlContext.Transforms.ProjectToPrincipalComponents(
                    outputColumnName: "PCAFeatures",
                    inputColumnName: "Features",
                    rank: config.NumberOfComponents));

            var model = await Task.Run(() => pipeline.Fit(dataView));
            var transformedData = model.Transform(dataView);

            var pcaResults = _mlContext.Data.CreateEnumerable<TransformedData>(
                transformedData, reuseRowObject: false).ToList();
            var pcaFeatures = pcaResults.Select(row => row.PCAFeatures).ToArray();

            var explainedVariance = CalculateExplainedVariance(pcaFeatures);

            var componentNames = Enumerable.Range(1, config.NumberOfComponents)
                                         .Select(i => $"PC{i}")
                                         .ToArray();

            _report.AppendLine("\nFeature Summary:");
            _report.AppendLine($"Original Features: {filteredCandidateFeatures.Length}");
            _report.AppendLine($"Components Created: {config.NumberOfComponents}");

            _report.AppendLine("\nFeatures Used:");
            foreach (var feature in filteredCandidateFeatures)
            {
                _report.AppendLine($"- {feature}");
            }

            _report.AppendLine("\nExplained Variance Ratios:");
            float cumulative = 0;
            for (int i = 0; i < explainedVariance.Length; i++)
            {
                cumulative += explainedVariance[i];
                _report.AppendLine($"  {componentNames[i]}: {explainedVariance[i]:P2} (Cumulative: {cumulative:P2})");
            }

            return (pcaFeatures, componentNames, _report.ToString());
        }

        private float[] CalculateExplainedVariance(float[][] pcaFeatures)
        {
            int n = pcaFeatures.Length;
            if (n == 0)
                return Array.Empty<float>();

            int components = pcaFeatures[0].Length;
            float[] variances = new float[components];
            float totalVariance = 0f;

            for (int j = 0; j < components; j++)
            {
                float mean = pcaFeatures.Average(row => row[j]);
                float variance = pcaFeatures.Sum(row => (row[j] - mean) * (row[j] - mean)) / (n - 1);
                variances[j] = variance;
                totalVariance += variance;
            }
            return variances.Select(v => v / totalVariance).ToArray();
        }

        private class InputData
        {
            [VectorType]
            public float[] Features { get; set; }
        }

        private class TransformedData
        {
            [VectorType]
            public float[] PCAFeatures { get; set; }
        }
    }
}