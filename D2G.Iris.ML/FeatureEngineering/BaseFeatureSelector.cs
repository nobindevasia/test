using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.FeatureEngineering
{
    public abstract class BaseFeatureSelector : IFeatureSelector
    {
        protected readonly MLContext _mlContext;
        protected readonly StringBuilder _report;

        protected BaseFeatureSelector(MLContext mlContext)
        {
            _mlContext = mlContext;
            _report = new StringBuilder();
        }

        public abstract Task<(float[][] features, string[] featureNames, string report)> SelectFeatures(
            List<Dictionary<string, object>> data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config);

     
        protected string[] FilterTargetField(string[] candidateFeatures, string targetField)
        {
            return candidateFeatures.Where(f => f != targetField).ToArray();
        }

        protected float[][] ConvertToFeatureMatrix(List<Dictionary<string, object>> data, string[] features)
        {
            return data.Select(dict =>
                features.Select(field =>
                    dict[field] == null ? 0.0f : Convert.ToSingle(dict[field])).ToArray()
            ).ToArray();
        }

        
        protected List<double> ExtractTargetValues(List<Dictionary<string, object>> data, string targetField)
        {
            return data
                .Where(row => row[targetField] != null)
                .Select(row => Convert.ToDouble(row[targetField]))
                .ToList();
        }

        
        protected List<double[]> ExtractFeatureValues(List<Dictionary<string, object>> data, string[] featureNames)
        {
            var result = new List<double[]>();
            foreach (var feature in featureNames)
            {
                var values = data.Select(row =>
                    row[feature] == null ? 0.0 : Convert.ToDouble(row[feature])).ToArray();
                result.Add(values);
            }
            return result;
        }

        protected static double EuclideanDistance(double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double diff = a[i] - b[i];
                sum += diff * diff;
            }
            return Math.Sqrt(sum);
        }

        protected static float EuclideanDistance(float[] a, float[] b)
        {
            float sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return MathF.Sqrt(sum);
        }

        protected string FormatSelectionSummary(string methodName, string[] originalFeatures, string[] selectedFeatures)
        {
            var summary = new StringBuilder();
            summary.AppendLine($"\n{methodName} Feature Selection Results:");
            summary.AppendLine("----------------------------------------");
            summary.AppendLine($"Original features: {originalFeatures.Length}");
            summary.AppendLine($"Selected features: {selectedFeatures.Length}");
            summary.AppendLine("\nSelected features:");

            foreach (var feature in selectedFeatures)
            {
                summary.AppendLine($"- {feature}");
            }

            return summary.ToString();
        }


        protected int[] FindKNearestNeighbors<T>(
            IList<T[]> features,
            T[] target,
            int excludeIndex,
            int k,
            Func<T[], T[], float> distanceFunc)
        {
            var distances = new List<(int index, float distance)>();

            for (int i = 0; i < features.Count; i++)
            {
                if (i == excludeIndex) continue;
                distances.Add((i, distanceFunc(features[i], target)));
            }

            return distances.OrderBy(x => x.distance)
                          .Take(k)
                          .Select(x => x.index)
                          .ToArray();

        }
    }
}