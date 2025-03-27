using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.FeatureEngineering;

namespace D2G.Iris.ML.FeatureEngineering
{
    public class NoFeatureSelector : BaseFeatureSelector
    {
        public NoFeatureSelector(MLContext mlContext) : base(mlContext)
        {
        }

        public override Task<(float[][] features, string[] featureNames, string report)> SelectFeatures(
            List<Dictionary<string, object>> data,
            string[] candidateFeatures,
            ModelType modelType,
            string targetField,
            FeatureEngineeringConfig config)
        {
            _report.Clear();
            _report.AppendLine("\nNo Feature Selection Applied");
            _report.AppendLine("----------------------------------------");
            _report.AppendLine($"Using all enabled features: {candidateFeatures.Length}");

            foreach (var feature in candidateFeatures)
            {
                _report.AppendLine($"- {feature}");
            }

            var filteredFeatures = FilterTargetField(candidateFeatures, targetField);
            var features = ConvertToFeatureMatrix(data, filteredFeatures);
            return Task.FromResult((features, filteredFeatures, _report.ToString()));
            
        }
    }
}