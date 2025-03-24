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
    public class AdasynDataBalancer : BaseDataBalancer
    {
        private const int DEFAULT_SYNTHETIC_BATCH_SIZE = 100;

        public override async Task<List<Dictionary<string, object>>> BalanceDataset(
            List<Dictionary<string, object>> data,
            string[] featureNames,
            DataBalancingConfig config,
            string targetField)
        {
            Console.WriteLine("Starting ADASYN data balancing...");
            var timer = System.Diagnostics.Stopwatch.StartNew();
            ValidateBalancingParameters(config);

            var random = new ThreadLocal<Random>(() => new Random(42));

            var (minorityClass, majorityClass) = SplitDataByClass(data, targetField);

            Console.WriteLine($"Minority class count: {minorityClass.Count}");
            Console.WriteLine($"Majority class count: {majorityClass.Count}");

            int undersampledMajorityCount = (int)(majorityClass.Count * config.UndersamplingRatio);
            var undersampledMajority = ShuffleMajorityClass(majorityClass, undersampledMajorityCount, random.Value);

          
            int targetMinorityCount = (int)(undersampledMajorityCount * config.MinorityToMajorityRatio);
            int syntheticCount = Math.Max(0, targetMinorityCount - minorityClass.Count);

            if (syntheticCount <= 0)
            {
                Console.WriteLine("No synthetic samples needed");
                return data;
            }

            Console.WriteLine($"Synthetic samples needed: {syntheticCount}");
            var minorityFeatures = ConvertToFeatureArrays(minorityClass, featureNames);
            var majorityFeatures = ConvertToFeatureArrays(majorityClass, featureNames);
            Console.WriteLine("Calculating density ratios...");
            var rValues = await CalculateRValues(minorityFeatures, majorityFeatures, config.KNeighbors);
            NormalizeArray(rValues);

            // Generate synthetic samples
            Console.WriteLine("Generating synthetic samples in batches...");
            var syntheticSamples = await GenerateSyntheticSamples(
                minorityClass,
                minorityFeatures,
                syntheticCount,
                rValues,
                featureNames,
                config.KNeighbors,
                DEFAULT_SYNTHETIC_BATCH_SIZE,
                random);

            // Combine all samples
            var result = new List<Dictionary<string, object>>();
            result.AddRange(minorityClass);
            result.AddRange(undersampledMajority);
            result.AddRange(syntheticSamples);

            timer.Stop();
            LogBalancingResults(
                "ADASYN",
                data.Count,
                minorityClass.Count,
                syntheticSamples.Count,
                majorityClass.Count,
                undersampledMajority.Count,
                result.Count,
                timer.ElapsedMilliseconds);

            return result;
        }

        private async Task<double[]> CalculateRValues(
            float[][] minorityFeatures,
            float[][] majorityFeatures,
            int k)
        {
            var rValues = new double[minorityFeatures.Length];
            var completedSamples = 0;

            await Task.Run(() =>
            {
                Parallel.For(0, minorityFeatures.Length, i =>
                {
                    var distances = new List<double>(majorityFeatures.Length);
                    var minorityVector = minorityFeatures[i];

                    foreach (var majorityVector in majorityFeatures)
                    {
                        distances.Add(EuclideanDistance(minorityVector, majorityVector));
                    }

                    var kNearestCount = distances.OrderBy(x => x).Take(k).Count();
                    rValues[i] = kNearestCount / (double)k;

                    var current = Interlocked.Increment(ref completedSamples);
                    if (current % 1000 == 0)
                    {
                        Console.WriteLine($"{current}/{minorityFeatures.Length} samples processed...");
                    }
                });
            });

            return rValues;
        }

        private async Task<List<Dictionary<string, object>>> GenerateSyntheticSamples(
            List<Dictionary<string, object>> minoritySamples,
            float[][] minorityFeatures,
            int syntheticCount,
            double[] rValues,
            string[] featureNames,
            int k,
            int batchSize,
            ThreadLocal<Random> random)
        {
            var syntheticSamples = new ConcurrentBag<Dictionary<string, object>>();
            var totalGenerated = 0;

            var nearestNeighborsCache = new ConcurrentDictionary<int, int[]>();

            for (int batchStart = 0; batchStart < syntheticCount; batchStart += batchSize)
            {
                int currentBatchSize = Math.Min(batchSize, syntheticCount - batchStart);

                await Task.Run(() =>
                {
                    Parallel.For(0, currentBatchSize, _ =>
                    {
                        int selectedIndex = SelectIndexBasedOnDistribution(rValues, random.Value);

                        var neighbors = nearestNeighborsCache.GetOrAdd(selectedIndex, idx =>
                            FindKNearestNeighbors(minorityFeatures, minorityFeatures[idx], idx, k));

                        var neighborIndex = neighbors[random.Value.Next(neighbors.Length)];

                        var synthetic = CreateSyntheticSample(
                            InterpolateFeatures(minorityFeatures[selectedIndex], minorityFeatures[neighborIndex], random.Value),
                            featureNames,
                            minoritySamples[0]["Label"]);

                        syntheticSamples.Add(synthetic);

                        var current = Interlocked.Increment(ref totalGenerated);
                        if (current % 10000 == 0)
                        {
                            Console.WriteLine($"{current}/{syntheticCount} synthetic samples generated...");
                        }
                    });
                });
            }

            return syntheticSamples.ToList();
        }
    }
}