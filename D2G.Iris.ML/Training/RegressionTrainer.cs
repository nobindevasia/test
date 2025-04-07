using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;
using D2G.Iris.ML.Training;

namespace D2G.Iris.ML.Training
{
    public class RegressionTrainer : BaseModelTrainer
    {
        public RegressionTrainer(MLContext mlContext, TrainerFactory trainerFactory)
            : base(mlContext, trainerFactory)
        {
        }

        public override async Task<IDataView> PrepareDataView(
            MLContext mlContext,
            ProcessedData processedData,
            string[] featureNames,
            string targetField)
        {
            var rows = processedData.Data.Select(dict => new RegressionRow
            {
                Features = featureNames.Select(field =>
                    dict[field] == null ? 0.0f : Convert.ToSingle(dict[field])).ToArray(),
                Label = Convert.ToSingle(dict[targetField])
            }).ToList();

            var schema = SchemaDefinition.Create(typeof(RegressionRow));
            schema["Features"].ColumnType = new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);

            return mlContext.Data.LoadFromEnumerable(rows, schema);
        }

        public override async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            var pipeline = GetBasePipeline(mlContext);
            var trainer = _trainerFactory.GetTrainer(config.ModelType, config.TrainingParameters);
            var trainingPipeline = pipeline.Append(trainer);

            var splitData = mlContext.Data.TrainTestSplit(dataView, testFraction: config.TrainingParameters.TestFraction);
            Console.WriteLine($"\n Starting regression model training using {config.TrainingParameters.Algorithm}...");

            var trainedModel = await Task.Run(() => trainingPipeline.Fit(splitData.TrainSet));

            var metrics = mlContext.Regression.Evaluate(trainedModel.Transform(splitData.TestSet));
            ConsoleHelper.PrintRegressionMetrics(config.TrainingParameters.Algorithm, metrics);

            var modelPath = $"Regression_{config.TrainingParameters.Algorithm}_Model.zip";
            mlContext.Model.Save(trainedModel, dataView.Schema, modelPath);
            Console.WriteLine($"Model saved to {modelPath}");

            await ModelHelper.CreateModelInfo<RegressionMetrics, float>(
                metrics,
                dataView,
                featureNames,
                config,
                processedData
            );

            return trainedModel;
        }

        private class RegressionRow
        {
            [VectorType]
            public float[] Features { get; set; }
            public float Label { get; set; }
        }
    }
}