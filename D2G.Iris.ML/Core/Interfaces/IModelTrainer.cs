using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IModelTrainer
    {
        Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData);

        Task<IDataView> PrepareDataView(
            MLContext mlContext,
            ProcessedData processedData,
            string[] featureNames,
            string targetField);
    }
}