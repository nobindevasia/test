using System.Collections.Generic;
using System.Threading.Tasks;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IDataBalancer
    {
        Task<List<Dictionary<string, object>>> BalanceDataset(
            List<Dictionary<string, object>> data,
            string[] featureNames,
            DataBalancingConfig config,
            string targetField);
    }
}