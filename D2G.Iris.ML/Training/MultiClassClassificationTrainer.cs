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
    public class MultiClassClassificationTrainer : BaseModelTrainer
    {
        public MultiClassClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
            : base(mlContext, trainerFactory)
        {
        }

        public override async Task<IDataView> PrepareDataView(
            MLContext mlContext,
            ProcessedData processedData,
            string[] featureNames,
            string targetField)
        {
            var rows = processedData.Data.Select(dict => new MulticlassRow
            {
                Features = featureNames.Select(field =>
                    dict[field] == null ? 0.0f : Convert.ToSingle(dict[field])).ToArray(),
                Label = Convert.ToUInt32(dict[targetField])
            }).ToList();

            var schema = SchemaDefinition.Create(typeof(MulticlassRow));
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
            var pipeline = mlContext.Transforms.NormalizeMinMax("Features")
                                   .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                                   .AppendCacheCheckpoint(mlContext);

            var trainer = _trainerFactory.GetTrainer(config.ModelType, config.TrainingParameters);
            var trainingPipeline = pipeline.Append(trainer)
                                 .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var splitData = mlContext.Data.TrainTestSplit(dataView, testFraction: config.TrainingParameters.TestFraction);
            Console.WriteLine($"Starting multiclass classification model training using {config.TrainingParameters.Algorithm}...");

            var trainedModel = await Task.Run(() => trainingPipeline.Fit(splitData.TrainSet));
 
            var metrics = mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(splitData.TestSet));
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            ConsoleHelper.PrintMultiClassClassificationMetrics(config.TrainingParameters.Algorithm, metrics);
           
            var modelPath = $"MulticlassClassification_{config.TrainingParameters.Algorithm}_Model.zip";
            mlContext.Model.Save(trainedModel, dataView.Schema, modelPath);
            Console.WriteLine($"Model saved to {modelPath}");
         
            await ModelHelper.CreateModelInfo<MulticlassClassificationMetrics, float>(
                metrics,
                dataView,
                featureNames,
                config,
                processedData
            );

            return trainedModel;
        }

        private class MulticlassRow
        {
            [VectorType]
            public float[] Features { get; set; }
            public uint Label { get; set; }
        }
    }
}