using System.Collections.Generic;
using D2G.Iris.ML.Core.Enums;


namespace D2G.Iris.ML.Core.Models
{
    public class ProcessedData
    {
        public List<Dictionary<string, object>> Data { get; set; }
        public string[] FeatureNames { get; set; }
        public int OriginalSampleCount { get; set; }
        public int BalancedSampleCount { get; set; }
        public string FeatureSelectionReport { get; set; }
        public FeatureSelectionMethod FeatureSelectionMethod { get; set; }
        public DataBalanceMethod DataBalancingMethod { get; set; }
        public float[][] Features { get; set; }
    }
}