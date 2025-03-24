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
    public class ForwardFeatureSelector : BaseFeatureSelector
    {
        public ForwardFeatureSelector(MLContext mlContext) : base(mlContext)
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
            _report.AppendLine("\nForward Feature Selection Results:");
            _report.AppendLine("----------------------------------------");

            try
            {
                var selectedFeatures = new List<string>();
                var filteredCandidateFeatures = FilterTargetField(candidateFeatures, targetField);
                var remainingFeatures = new List<string>(filteredCandidateFeatures);
                double currentBestScore = 0;

                while (remainingFeatures.Count > 0 && selectedFeatures.Count < config.MaxFeatures)
                {
                    var bestFeature = "";
                    var bestScore = currentBestScore;

                    foreach (var feature in remainingFeatures)
                    {
                        var featuresToTry = selectedFeatures.Concat(new[] { feature }).ToArray();
                        var score = await EvaluateFeatures(data, featuresToTry, modelType, targetField);

                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestFeature = feature;
                        }
                    }

                    if (bestFeature != "" && (bestScore - currentBestScore) >= config.MinImprovement)
                    {
                        selectedFeatures.Add(bestFeature);
                        remainingFeatures.Remove(bestFeature);
                        currentBestScore = bestScore;
                    }
                    else
                    {
                        break;
                    }
                }

                _report.AppendLine($"- Selected features: {selectedFeatures.Count}");
                _report.AppendLine($"- Final score: {currentBestScore:F4}");
                _report.AppendLine("\nSelected features in order:");

                foreach (var feature in selectedFeatures)
                {
                    _report.AppendLine($"- {feature}");
                }

                var features = ConvertToFeatureMatrix(data, selectedFeatures.ToArray());
                return (features, selectedFeatures.ToArray(), _report.ToString());
            }
            catch (Exception ex)
            {
                _report.AppendLine($"Error during forward selection: {ex.Message}");
                return (new float[0][], Array.Empty<string>(), _report.ToString());
            }
        }

        private async Task<double> EvaluateFeatures(
            List<Dictionary<string, object>> data,
            string[] features,
            ModelType modelType,
            string targetField)
        {
            try
            {
                switch (modelType)
                {
                    case ModelType.BinaryClassification:
                        return await EvaluateBinary(data, features, targetField);
                    case ModelType.MultiClassClassification:
                        return await EvaluateMulticlass(data, features, targetField);
                    case ModelType.Regression:
                        return await EvaluateRegression(data, features, targetField);
                    default:
                        throw new ArgumentException($"Unsupported model type: {modelType}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error evaluating features: {ex.Message}");
                return 0.0;
            }
        }

        private async Task<double> EvaluateBinary(
            List<Dictionary<string, object>> data,
            string[] features,
            string targetField)
        {
            var rows = data.Select(dict => new BinaryData
            {
                Features = features.Select(field =>
                    dict[field] == null ? 0.0f : Convert.ToSingle(dict[field])).ToArray(),
                Label = Convert.ToBoolean(dict[targetField])
            }).ToList();

            var schema = SchemaDefinition.Create(typeof(BinaryData));
            schema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, features.Length);

            var dataView = _mlContext.Data.LoadFromEnumerable(rows, schema);
            var split = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var pipeline = _mlContext.Transforms.NormalizeMinMax("Features")
                .Append(_mlContext.BinaryClassification.Trainers.FastTree(
                    labelColumnName: "Label",
                    featureColumnName: "Features"
                ));

            var model = await Task.Run(() => pipeline.Fit(split.TrainSet));
            var metrics = _mlContext.BinaryClassification.Evaluate(model.Transform(split.TestSet));
            return metrics.AreaUnderRocCurve;
        }

        private async Task<double> EvaluateMulticlass(
            List<Dictionary<string, object>> data,
            string[] features,
            string targetField)
        {
            var rows = data.Select(dict => new MulticlassData
            {
                Features = features.Select(field =>
                    dict[field] == null ? 0.0f : Convert.ToSingle(dict[field])).ToArray(),
                Label = Convert.ToUInt32(dict[targetField])
            }).ToList();

            var schema = SchemaDefinition.Create(typeof(MulticlassData));
            schema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, features.Length);

            var dataView = _mlContext.Data.LoadFromEnumerable(rows, schema);
            var split = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var pipeline = _mlContext.Transforms.NormalizeMinMax("Features")
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(_mlContext.MulticlassClassification.Trainers.LightGbm(
                    labelColumnName: "Label",
                    featureColumnName: "Features"
                ))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = await Task.Run(() => pipeline.Fit(split.TrainSet));
            var metrics = _mlContext.MulticlassClassification.Evaluate(model.Transform(split.TestSet));
            return metrics.MacroAccuracy;
        }

        private async Task<double> EvaluateRegression(
            List<Dictionary<string, object>> data,
            string[] features,
            string targetField)
        {
            var rows = data.Select(dict => new RegressionData
            {
                Features = features.Select(field =>
                    dict[field] == null ? 0.0f : Convert.ToSingle(dict[field])).ToArray(),
                Label = Convert.ToSingle(dict[targetField])
            }).ToList();

            var schema = SchemaDefinition.Create(typeof(RegressionData));
            schema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, features.Length);

            var dataView = _mlContext.Data.LoadFromEnumerable(rows, schema);
            var split = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            var pipeline = _mlContext.Transforms.NormalizeMinMax("Features")
                .Append(_mlContext.Regression.Trainers.FastTree(
                    labelColumnName: "Label",
                    featureColumnName: "Features"
                ));

            var model = await Task.Run(() => pipeline.Fit(split.TrainSet));
            var metrics = _mlContext.Regression.Evaluate(model.Transform(split.TestSet));
            return metrics.RSquared;
        }

        private class BinaryData
        {
            [VectorType]
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }

        private class MulticlassData
        {
            [VectorType]
            public float[] Features { get; set; }
            public uint Label { get; set; }
        }

        private class RegressionData
        {
            [VectorType]
            public float[] Features { get; set; }
            public float Label { get; set; }
        }
    }
}