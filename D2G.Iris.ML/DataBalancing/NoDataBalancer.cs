using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.DataBalancing;

namespace D2G.Iris.ML.DataBalancing
{
    public class NoDataBalancer : BaseDataBalancer
    {
        public override Task<List<Dictionary<string, object>>> BalanceDataset(
            List<Dictionary<string, object>> data,
            string[] featureNames,
            DataBalancingConfig config,
            string targetField)
        {
            Console.WriteLine("No data balancing applied - returning original dataset");
            return Task.FromResult(data);
        }
    }
}