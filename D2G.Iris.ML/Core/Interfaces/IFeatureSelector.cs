using Microsoft.ML;
using System.Collections.Generic;
using System.Threading.Tasks;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Enums;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IFeatureSelector
    {
        Task<(float[][] features, string[] featureNames, string report)> SelectFeatures(
            List<Dictionary<string, object>> data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config);
    }
}