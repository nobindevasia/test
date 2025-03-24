using Microsoft.ML;
using System.Collections.Generic;
using System.Threading.Tasks;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Enums;

namespace D2G.Iris.ML.Core.Interfaces
{
    public interface IDataLoader
    {
        IEnumerable<Dictionary<string, object>> LoadDataFromSql(
            string connectionString,
            string tableName,
            IEnumerable<string> fields,
            ModelType modelType,
            string targetField,
            string whereSyntax = "");

        IDataView ConvertToDataView(
            MLContext mlContext,
            IEnumerable<Dictionary<string, object>> data,
            string[] features,
            ModelType modelType,
            string targetField);
    }
}