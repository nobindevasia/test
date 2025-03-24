using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.DataBalancing
{
    public abstract class BaseDataBalancer : IDataBalancer
    {
        public abstract Task<List<Dictionary<string, object>>> BalanceDataset(
            List<Dictionary<string, object>> data,
            string[] featureNames,
            DataBalancingConfig config,
            string targetField);

       
        protected void ValidateBalancingParameters(DataBalancingConfig config)
        {
            if (config.UndersamplingRatio <= 0 || config.UndersamplingRatio > 1)
                throw new ArgumentException("Undersampling ratio must be between 0 and 1");

            if (config.MinorityToMajorityRatio <= 0 || config.MinorityToMajorityRatio > 1)
                throw new ArgumentException("Minority to majority ratio must be between 0 and 1");

            if (config.KNeighbors < 1)
                throw new ArgumentException("K should be greater than 0");
        }

     
        protected (List<Dictionary<string, object>> minority, List<Dictionary<string, object>> majority)
            SplitDataByClass(List<Dictionary<string, object>> data, string targetField)
        {
            var groupedData = data.GroupBy(x => Convert.ToBoolean(x[targetField]))
                                .OrderBy(g => g.Count())
                                .ToList();

            if (groupedData.Count != 2)
                throw new ArgumentException("Data balancing requires exactly two classes");

            return (groupedData[0].ToList(), groupedData[1].ToList());
        }

      
        protected List<Dictionary<string, object>> ShuffleMajorityClass(
            List<Dictionary<string, object>> samples,
            int targetCount,
            Random random)
        {
            var indices = new int[samples.Count];
            for (int i = 0; i < indices.Length; i++)
                indices[i] = i;

            // Fisher-Yates shuffle
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            return indices.Take(targetCount)
                         .Select(i => samples[i])
                         .ToList();
        }

      
        protected float[][] ConvertToFeatureArrays(
            List<Dictionary<string, object>> samples,
            string[] featureNames)
        {
            return samples.Select(x =>
                featureNames.Select(f => Convert.ToSingle(x[f])).ToArray()
            ).ToArray();
        }

      
        protected int[] FindKNearestNeighbors(IList<float[]> features, float[] target, int excludeIndex, int k)
        {
            var distances = new List<(int index, float distance)>();

            for (int i = 0; i < features.Count; i++)
            {
                if (i == excludeIndex) continue;
                distances.Add((i, EuclideanDistance(features[i], target)));
            }

            return distances.OrderBy(x => x.distance)
                          .Take(k)
                          .Select(x => x.index)
                          .ToArray();
        }

        
        protected float EuclideanDistance(float[] a, float[] b)
        {
            float sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return MathF.Sqrt(sum);
        }

       
        protected float[] InterpolateFeatures(float[] a, float[] b, Random random)
        {
            float ratio = (float)random.NextDouble();
            var result = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + ratio * (b[i] - a[i]);
            }

            return result;
        }

    
        protected Dictionary<string, object> CreateSyntheticSample(
            float[] features,
            string[] featureNames,
            object label)
        {
            var sample = new Dictionary<string, object>();
            for (int i = 0; i < featureNames.Length; i++)
            {
                sample[featureNames[i]] = features[i];
            }
            sample["Label"] = label;
            return sample;
        }


        protected void NormalizeArray(double[] values)
        {
            double sum = values.Sum();
            if (sum > 0)
            {
                var factor = 1.0 / sum;
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] *= factor;
                }
            }
            else
            {
                var value = 1.0 / values.Length;
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = value;
                }
            }
        }

        protected int SelectIndexBasedOnDistribution(double[] distribution, Random random)
        {
            double rand = random.NextDouble();
            double cumulativeProb = 0;

            for (int i = 0; i < distribution.Length; i++)
            {
                cumulativeProb += distribution[i];
                if (rand <= cumulativeProb)
                    return i;
            }

            return distribution.Length - 1;
        }

        protected void LogBalancingResults(
            string methodName,
            int totalSamples,
            int originalMinority,
            int synthetic,
            int originalMajority,
            int undersampledMajority,
            int finalCount,
            long elapsedMs = 0)
        {
            Console.WriteLine($"\n{methodName} Balancing Results" + (elapsedMs > 0 ? $" ({elapsedMs}ms)" : ""));
            Console.WriteLine($"Original dataset: {totalSamples:N0} samples");
            Console.WriteLine($"Minority class (original): {originalMinority:N0} samples");
            Console.WriteLine($"Minority class (after balancing): {(originalMinority + synthetic):N0} samples");
            Console.WriteLine($"Majority class (original): {originalMajority:N0} samples");
            Console.WriteLine($"Majority class (after undersampling): {undersampledMajority:N0} samples");
            Console.WriteLine($"Balanced dataset: {finalCount:N0} samples");
            Console.WriteLine($"Final ratio: {(originalMinority + synthetic) / (float)undersampledMajority:F2}");
        }
    }
}