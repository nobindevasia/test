using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.DataBalancing;

namespace D2G.Iris.ML.DataBalancing
{
    public class SmoteDataBalancer : BaseDataBalancer
    {
        public override async Task<List<Dictionary<string, object>>> BalanceDataset(
            List<Dictionary<string, object>> data,
            string[] featureNames,
            DataBalancingConfig config,
            string targetField)
        {
            if (config.UndersamplingRatio <= 0 || config.UndersamplingRatio > 1)
                throw new ArgumentException("Undersampling ratio must be between 0 and 1");

            if (config.MinorityToMajorityRatio <= 0 || config.MinorityToMajorityRatio > 1)
                throw new ArgumentException("Minority to majority ratio must be between 0 and 1");

            if (config.KNeighbors < 1)
                throw new ArgumentException("K should be greater than 0");

            Console.WriteLine("=============== Balancing Dataset with SMOTE ===============");
            var timer = System.Diagnostics.Stopwatch.StartNew();
            var random = new Random(42);

            // Group data by label
            var groupedData = data.GroupBy(x => Convert.ToBoolean(x[targetField]))
                                .OrderBy(g => g.Count())
                                .ToList();

            if (groupedData.Count != 2)
                throw new ArgumentException("Dataset must contain exactly two classes");

            var minorityClass = groupedData[0].ToList();
            var majorityClass = groupedData[1].ToList();

            // Undersample majority class
            int undersampledMajorityCount = (int)(majorityClass.Count * config.UndersamplingRatio);
            var undersampledMajority = ShuffleMajorityClass(majorityClass, undersampledMajorityCount, random);

            // Calculate synthetic samples needed
            int targetMinorityCount = (int)(undersampledMajorityCount * config.MinorityToMajorityRatio);
            int syntheticCount = Math.Max(0, targetMinorityCount - minorityClass.Count);

            var result = new List<Dictionary<string, object>>();
            result.AddRange(undersampledMajority);
            result.AddRange(minorityClass);

            if (syntheticCount > 0)
            {
                Console.WriteLine($"Generating {syntheticCount} synthetic samples using k={config.KNeighbors}");
                var syntheticSamples = await GenerateSyntheticSamples(
                    minorityClass,
                    syntheticCount,
                    featureNames,
                    config.KNeighbors,
                    random);
                result.AddRange(syntheticSamples);
            }

            timer.Stop();
            LogResults(
                data.Count,
                minorityClass.Count,
                syntheticCount,
                majorityClass.Count,
                undersampledMajority.Count,
                result.Count,
                timer.ElapsedMilliseconds);

            return result;
        }

        private static List<Dictionary<string, object>> ShuffleMajorityClass(
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

        private static async Task<List<Dictionary<string, object>>> GenerateSyntheticSamples(
            List<Dictionary<string, object>> minoritySamples,
            int syntheticCount,
            string[] featureNames,
            int k,
            Random random)
        {
            // Convert samples to feature arrays for faster processing
            var features = minoritySamples.Select(x =>
                featureNames.Select(f => Convert.ToSingle(x[f])).ToArray()
            ).ToList();

            // Calculate samples per minority instance
            int samplesPerInstance = (int)Math.Ceiling((double)syntheticCount / minoritySamples.Count);
            var syntheticSamples = new ConcurrentBag<Dictionary<string, object>>();

            // Pre-compute nearest neighbors for each sample
            var nearestNeighbors = new Dictionary<int, int[]>();
            for (int i = 0; i < features.Count; i++)
            {
                nearestNeighbors[i] = FindKNearestNeighbors(features, features[i], i, k);
            }

            await Task.Run(() =>
            {
                Parallel.For(0, minoritySamples.Count, i =>
                {
                    var localRandom = new Random(random.Next());
                    var baseFeatures = features[i];
                    var neighbors = nearestNeighbors[i];

                    for (int j = 0; j < samplesPerInstance && syntheticSamples.Count < syntheticCount; j++)
                    {
                        var neighborIdx = neighbors[localRandom.Next(neighbors.Length)];
                        var neighborFeatures = features[neighborIdx];

                        var syntheticFeatures = InterpolateFeatures(baseFeatures, neighborFeatures, localRandom);
                        var syntheticSample = CreateSyntheticSample(syntheticFeatures, featureNames, minoritySamples[0]["Label"]);

                        syntheticSamples.Add(syntheticSample);
                    }
                });
            });

            return syntheticSamples.Take(syntheticCount).ToList();
        }

        private static int[] FindKNearestNeighbors(List<float[]> features, float[] target, int excludeIndex, int k)
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

        private static float EuclideanDistance(float[] a, float[] b)
        {
            float sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                float diff = a[i] - b[i];
                sum += diff * diff;
            }
            return MathF.Sqrt(sum);
        }

        private static float[] InterpolateFeatures(float[] a, float[] b, Random random)
        {
            float ratio = (float)random.NextDouble();
            var result = new float[a.Length];

            for (int i = 0; i < a.Length; i++)
            {
                result[i] = a[i] + ratio * (b[i] - a[i]);
            }

            return result;
        }

        private static Dictionary<string, object> CreateSyntheticSample(
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

        private static void LogResults(
            int totalSamples,
            int originalMinority,
            int synthetic,
            int originalMajority,
            int undersampledMajority,
            int finalCount,
            long elapsedMs)
        {
            Console.WriteLine($"\nBalancing Results ({elapsedMs}ms):");
            Console.WriteLine($"Original dataset: {totalSamples:N0} samples");
            Console.WriteLine($"Minority class (original): {originalMinority:N0} samples");
            Console.WriteLine($"Minority class (after SMOTE): {(originalMinority + synthetic):N0} samples");
            Console.WriteLine($"Majority class (original): {originalMajority:N0} samples");
            Console.WriteLine($"Majority class (after undersampling): {undersampledMajority:N0} samples");
            Console.WriteLine($"Balanced dataset: {finalCount:N0} samples");
            Console.WriteLine($"Final ratio: {(originalMinority + synthetic) / (float)undersampledMajority:F2}");
        }
    }
}