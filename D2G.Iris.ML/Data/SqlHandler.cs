using System;
using System.Collections.Generic;
using System.Data;
using Microsoft.Data.SqlClient;
using System.Linq;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Data
{
    public class SqlHandler : ISqlHandler
    {
        private SqlConnectionStringBuilder _builder;
        private readonly string _tableName;

        public SqlHandler(string tableName)
        {
            _tableName = tableName;
        }

        public void Connect(DatabaseConfig dbConfig)
        {
            _builder = new SqlConnectionStringBuilder()
            {
                DataSource = dbConfig.Server,
                InitialCatalog = dbConfig.Database,
                IntegratedSecurity = true,
                Pooling = true,
                TrustServerCertificate = true,
                ConnectTimeout = 60
            };
        }

        public string GetConnectionString()
        {
            if (_builder == null)
                throw new InvalidOperationException("Database connection not initialized");
            return _builder.ConnectionString;
        }

        public void SaveProcessedDataToTable(
            string tableName,
            List<Dictionary<string, object>> processedData,
            string[] featureNames,
            string targetField,
            ModelType modelType)
        {
            var finalFeatureNames = featureNames.Where(f => f != targetField).ToArray();

            using (var connection = new SqlConnection(GetConnectionString()))
            {
                connection.Open();

                string targetColumnType = modelType switch
                {
                    ModelType.BinaryClassification => "bit",
                    ModelType.MultiClassClassification => "int",
                    ModelType.Regression => "float",
                    _ => "float"
                };

                string dropAndCreateTableSql = $@"
                IF EXISTS (SELECT * FROM sys.tables WHERE name = '{tableName.Split('.').Last()}' 
                          AND SCHEMA_NAME(schema_id) = '{tableName.Split('.').First()}')
                BEGIN
                    DROP TABLE {tableName}
                END

                CREATE TABLE {tableName} (
                    {string.Join(",\n                    ", finalFeatureNames.Select(f => $"[{f}] float"))},
                    [{targetField}] {targetColumnType},
                    ProcessedDateTime datetime DEFAULT GETDATE()
                )";

                using (var command = new SqlCommand(dropAndCreateTableSql, connection))
                {
                    try
                    {
                        command.ExecuteNonQuery();
                        Console.WriteLine("\n");
                        Console.WriteLine($"Table {tableName} dropped and recreated successfully");
                    }
                    catch (SqlException ex)
                    {
                        Console.WriteLine($"Error recreating table: {ex.Message}");
                        throw;
                    }
                }

                using (var bulkCopy = new SqlBulkCopy(connection))
                {
                    bulkCopy.DestinationTableName = tableName;
                    bulkCopy.BatchSize = 1000;
                    bulkCopy.BulkCopyTimeout = 600;

                    var dataTable = new DataTable();

                    foreach (var feature in finalFeatureNames)
                    {
                        dataTable.Columns.Add(feature, typeof(float));
                    }

                    var targetColumnClrType = modelType switch
                    {
                        ModelType.BinaryClassification => typeof(bool),
                        ModelType.MultiClassClassification => typeof(int),
                        ModelType.Regression => typeof(float),
                        _ => typeof(float)
                    };
                    dataTable.Columns.Add(targetField, targetColumnClrType);

                    foreach (var row in processedData)
                    {
                        var dataRow = dataTable.NewRow();
                        foreach (var feature in finalFeatureNames)
                        {
                            dataRow[feature] = Convert.ToSingle(row[feature]);
                        }
                        dataRow[targetField] = row[targetField];
                        dataTable.Rows.Add(dataRow);
                    }

                    foreach (var column in dataTable.Columns)
                    {
                        bulkCopy.ColumnMappings.Add(column.ToString(), column.ToString());
                    }
                    
                    try
                    {
                        bulkCopy.WriteToServer(dataTable);
                        Console.WriteLine($"Successfully inserted {dataTable.Rows.Count} rows into {tableName}");
                    }
                    catch (Exception ex)
                    {
                        throw new Exception($"Failed to insert data into table {tableName}: {ex.Message}", ex);
                    }
                }
            }
        }
    }
}