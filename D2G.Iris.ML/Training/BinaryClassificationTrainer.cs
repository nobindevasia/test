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
    public class BinaryClassificationTrainer : BaseModelTrainer
    {
        public BinaryClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
            : base(mlContext, trainerFactory)
        {
        }

        public override async Task<IDataView> PrepareDataView(
            MLContext mlContext,
            ProcessedData processedData,
            string[] featureNames,
            string targetField)
        {
            var rows = processedData.Data.Select(dict => new BinaryRow
            {
                Features = featureNames.Select(field =>
                    dict[field] == null ? 0.0f : Convert.ToSingle(dict[field])).ToArray(),
                Label = Convert.ToBoolean(dict[targetField])
            }).ToList();
            var schema = SchemaDefinition.Create(typeof(BinaryRow));
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

            var trainingPipeline = pipeline
                .Append(trainer)
                .Append(mlContext.Transforms.CopyColumns("Probability", "Score"));

            var splitData = mlContext.Data.TrainTestSplit(dataView, testFraction: config.TrainingParameters.TestFraction);
            Console.WriteLine($"\nStarting binary classification model training using {config.TrainingParameters.Algorithm}...");

            // Train 
            var trainedModel = await Task.Run(() => trainingPipeline.Fit(splitData.TrainSet));

            // Evaluate
            var metrics = mlContext.BinaryClassification.Evaluate(
                trainedModel.Transform(splitData.TestSet),
                labelColumnName: "Label",
                predictedLabelColumnName: "PredictedLabel",
                scoreColumnName: "Probability");

            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            ConsoleHelper.PrintBinaryClassificationMetrics(config.TrainingParameters.Algorithm, metrics);

            // Save
            var modelPath = $"BinaryClassification_{config.TrainingParameters.Algorithm}_Model.zip";
            mlContext.Model.Save(trainedModel, dataView.Schema, modelPath);
            Console.WriteLine($"Model saved to {modelPath}");

            await ModelHelper.CreateModelInfo<BinaryClassificationMetrics, float>(
                metrics,
                dataView,
                featureNames,
                config,
                processedData
            );
            return trainedModel;
        }

        private class BinaryRow
        {
            [VectorType]
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }
    }
}