# SPDX-FileCopyrightText: 2024-present Brandon Rose <rose.brandon.m@gmail.com>
#
# SPDX-License-Identifier: MIT
import traceback
import re
import json
import logging
from typing import TYPE_CHECKING, Dict, List, Tuple, Type
from archytas.tool_utils import AgentRef, LoopControllerRef, ReactContextRef, tool
from archytas.exceptions import ExecutionError
from beaker_kernel.lib import BeakerAgent

if TYPE_CHECKING:
    from beaker_kernel.kernel import BeakerKernel


JSON_OUTPUT = False
PRINT_OUTPUT = True

ANSI_ESCAPE = re.compile(r'\x1b\[.*?m')
PROCEDURE_LOGGER_TEMPLATE_PATTERN = (
    r"^\s*" # Optional leading whitespace
    r"[\d\s,:-]+?"  # Match the timestamp (e.g., "2025-11-02 16:38:00,123")
    r"\s+-\s+" # Separator
    r"[\w.]+" # The logger name
    r"\s+-\s+" # Separator
    # Capture the log level, but only if it's WARNING or higher
    r"{level_part}"
    r"\s+-\s+" # Separator
    # Capture the rest of the line as the message
    r"(?P<message>.*)$"
)

class BaseAgent(BeakerAgent):
    """
    Replace original tool from ReActAgent. The name and docstring of this tool often causes
    agents to misunderstand its purpose and misuse it.
    """
    @tool(autosummarize=True)
    def retrieve_summarized_messages_of_summary(
        self,
        summary_message_uuid: str,
        agent_ref: AgentRef,
    ) -> str:
        """
        Temporarily retrieves and rehydrates the full, raw messages previously condensed into a summary message.
        The contents will be available during the current ReAct loop, but will be removed once the current loop is
        finished.

        Args:
            summary_message_uuid (str): The UUIDv4 identifier of the summary message from which you want to recover the
                    full, unsummarized messages.

        Returns:
            str: The full contents of the requested previously summarized messages.
        """
        return super().retrieve_summarized_messages_of_summary(summary_message_uuid, agent_ref)


class BdfPzAgent(BaseAgent):
    """
    You are a helpful agent that is intended to assist users in using Palimpzest, a
    declarative system for optimizing AI workloads.

    """

    def _parse_procedure_logs(
        self,
        stderr_list: list[str],
        min_level=logging.WARNING,
        parse_log_message=False
    ) -> dict[str, list[str]]:
        log_level_map = logging.getLevelNamesMapping()
        captured_level_names = [level_name for (level_name, level_value) in log_level_map.items() if level_value >= min_level]
        level_part = f"(?P<level>{'|'.join(captured_level_names)})"
        log_pattern = re.compile(PROCEDURE_LOGGER_TEMPLATE_PATTERN.format(level_part=level_part))
        
        parsed_logs = { level_name : [] for level_name in log_level_map.keys() }
        for line in stderr_list:
            match = log_pattern.match(line.strip())
            if match:
                log_level = match.group("level")
                log_message = match.group("message")
                parsed_logs[log_level].append(log_message if parse_log_message else line)
        return { level: level_logs for (level, level_logs) in parsed_logs.items() if len(level_logs) > 0 }

    # NOTE: Unless @tool reraises when result["error"] is defined, the agent will be in
    # the dark as to if something went wrong in the tool execution.
    def handle_response(self, result: dict) -> None:
        error = result.get("error")
        if error:
            error_traceback = "\n".join([ANSI_ESCAPE.sub("", line) for line in error["traceback"]])
            raise ExecutionError(error_traceback)
        
        out = result.get("return")
        parsed_logs = self._parse_procedure_logs(result.get("stderr_list"), min_level=logging.WARNING, parse_log_message=True)
        for log_level in parsed_logs:
            level_logs = parsed_logs[log_level]
            out += f"\n\n[TOOL {log_level} LOGS]:\n" + "\n".join(f"- { log }" for log in level_logs)
        
        return out

    async def auto_context(self):
        return """You are an assistant that is intended to assist users in using Palimpzest.
        Try to identify all of the steps needed, and all of the tools. Assume the user wants to do all of the steps at once.

        If the user asks to extract something from a set of documents, you can use Palimpzest to do this. First, generate a schema for the extraction. Then, if necessary filter the data to only include the relevant documents. Next, convert the dataset to the schema that was generated. Finally, execute the workload to extract the information from the dataset.
        You may need to use multiple tools to accomplish this, including the ability to register datasets, setting the input source, filtering datasets,
        convert datasets, generating schemas, and executing workloads.

        Make sure you understand all the steps needed to complete the task. Try to run all of the steps at once.

        Always use `run_code` for tasks that other tools do not enable you to accomplish. 
        """

    @tool()
    async def register_dataset(self, path: str, name: str, agent: AgentRef) -> str:
        """
        This function registers a dataset with Palimpzest. It takes a path to a file or directory
        and a name for the dataset. The dataset will be registered and made available for use in
        subsequent operations.

        Args:
            path (str): The path to the file or directory to register as a dataset.
            name (str): The name to give to the registered dataset. If not explicitly set, the name of the file or directory will be used.

        Returns:
            str: A message indicating the result of the registration process.
        """

        code = agent.context.get_code("register_dataset", {"path": path, "name": name})
        response = await agent.context.evaluate(code)
        return self.handle_response(response)

    @tool()
    async def unregister_dataset(self, dataset_name: str, agent: AgentRef) -> str:
        """
        This function unregisters a dataset with Palimpzest. It takes a dataset name and unregisters the dataset. The dataset will be unregistered and made
        unavailable for use in subsequent operations.

        Args:
            dataset_name (str): The name of the dataset to unregister.

        Returns:
            str: A message indicating the result of the unregistration process.
        """

        code = agent.context.get_code(
            "unregister_dataset", {"dataset_name": dataset_name}
        )
        if PRINT_OUTPUT:
            print(code)
        response = await agent.context.evaluate(code)
        return self.handle_response(response)

    @tool()
    async def list_datasets(self, agent: AgentRef) -> str:
        """
        This function lists all available datasets in the system.

        Returns:
            str: A table of the datasets in the system (path, name, file_count).
        """

        code = agent.context.get_code("list_datasets", {})
        if PRINT_OUTPUT:
            print(code)

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )

            return self.handle_response(result)

    @tool()
    async def retrieve_dataset(self, dataset_name: str, agent: AgentRef) -> list[str]:
        """
        This function lists the available files within a given dataset.

        Args:
            dataset_name (str): The name of the dataset to retrieve.

        Returns:
            list[str]: a list of the item identifiers (e.g., filenames, keys, etc...) available to the user in the given dataset.
        """

        code = agent.context.get_code(
            "retrieve_dataset",
            {"dataset_name": dataset_name},
        )
        if PRINT_OUTPUT:
            print(code)

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )

            return self.handle_response(result)

    @tool()
    async def create_schema(
        self,
        schema_name: str,
        field_names: list[str],
        field_descriptions: list[str],
        field_types: list[str],
        agent: AgentRef,
        loop: LoopControllerRef
    ) -> str:
        """
        This function takes in a set of fields to be used to generate an extraction schema.
        Typically it is called when users want to extract some piece of information from a set of documents.
        After the schema is created, the input dataset should be converted to the new schema.
        This should be used when the user is interested in generating a new type of extraction schema. For example, let's say the user is interested in extracting parameter values from a set of scientific papers. The user can define the fields of the schema to be used for the extraction.
        In this case the schema name might be `Parameter` and the field information is passed in via three lists which must be constructed in proper order. For example, for parameter extractions the fields may be `name`, `value`, `unit`, `source`, etc.
        You should provide a description for each field as well as the type of the field ("str", "int", "float", "bool"). These have to be in the same order as you provide the field names. Field names should not have spaces or special characters, but can have underscores.

        Args:
            schema_name (str): the name of the schema to add
            field_names (list[str]): a list of field names
            field_descriptions (list[str]): a list of field descriptions
            field_types (list[str]): a list of strings representing Python types for the fields. Each element must be one of the literals "str", "bool", "int", or "float".

        Returns:
            str: the name of the new schema that was created
        """

        code = agent.context.get_code(
            "create_schema",
            {
                "schema_name": schema_name,
                "field_names": field_names,
                "field_descriptions": field_descriptions,
                "field_types": field_types,
            },
        )
        if PRINT_OUTPUT:
            print(code)

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )

            return self.handle_response(result)

    @tool()
    async def filter_data(
        self,
        input_dataset: str,
        filter_expression: str,
        agent: AgentRef,
        loop: LoopControllerRef,
        computed_from: list[str] | None = None
    ) -> str:
        """
        This function generates a filtered dataset given an input dataset and a filtering expression.
        The filter expression is a string that describes a condition that has to be satisfied for each of the data item in the dataset.
        The computed_from field can be used to filter against a subset of each data item's fields rather than using the entirety of each item. Use None to filter against all fields.
        If there is ANY ambiguity regarding which fields the filter should be computed using, THEN THIS SHOULD BE `NONE`, which will filter against all of each item's fields.
        For example if a user is interested in a dataset of scientific papers and wants to only keep papers that are published in the year 2022, the filter expression might be "The papers is published in 2022". If a publication_date field exists on the dataset, then you might specify computed_from=["publication_date"].

        Args:
            input_dataset (str): The input Dataset to use for the filtering.
            filter_expression (str): A string that describes a condition in natural language that can be used to filter out data points within a collection.
            computed_from (list[str], optional): A subset of input field(s) to apply the filter against. Defaults to None, which will filter against all of the item's fields. 
            
        Returns:
            str: returns a new dataset corresponding to the filtered input dataset on line 1, and a schema for its fields on line 2.
        
        You should show the user the filter you used and what fields it is computed from (all fields if None) after this function runs.
        """

        code = agent.context.get_code(
            "filter_data",
            {
                "input_dataset": input_dataset,
                "filter_expression": filter_expression,
                "computed_from": computed_from if computed_from else None
            },
        )
        if PRINT_OUTPUT:
            print(code)

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )

            return self.handle_response(result)

    @tool
    async def convert_dataset(
        self,
        input_dataset: str,
        schema_name: str,
        cardinality: str,
        agent: AgentRef,
        loop: LoopControllerRef,
        computed_from: list[str] | None = None
    ) -> str:
        """
        This function creates an output dataset by augmenting an input dataset with a specific schema.
        The function has to be used to extract any information from a collection of input documents.
        The function is typically needed before executing a workload, to apply a generated schema to an existing dataset.
        If the schema object can be extracted multiple times from a single object of the input dataset, the cardinality should be set to "one_to_many". If the schema can only be extracted once from a single object of the input dataset, the cardinality should be set to "one_to_one".
        The computed_from field can be used to specify a subset of the dataset's fields to use to compute the output schema, rather than computing the schema against the entirety of each input item.
        If there is ambiguity regarding which fields are required to compute the schema, then this should be None. Leaving as None will require more computational effort, which is OK.
        For example if a user wants to extract the titles for a dataset of scientific papers, the schema might be a TitleSchema, and the cardinality would be one_to_one. There would be no computed_from specified.
        For example, if a user wants to translate the abstracts for a dataset of scientific papers, and an "abstract" field already exists on the dataset, then the schema might be a TranslateAbstract, the cardinality would be one_to_one, and the computed_from field would be ["abstract"].


        Args:
            input_dataset (str): An existing object of type dataset to use for conversion.
            schema_name (str): The name of a schema from the ones existing in the system that describes the object of the new converted dataset.
            cardinality (str): The cardinality of the conversion. Either "one_to_one" or "one_to_many".
            computed_from (list[str], optional): A subset of input field(s) used to compute the output schema. This should be used to save time when the fields required to compute the schema are already defined on the input dataset. Defaults to None, which will use all fields to compute the output schema. 
            
        Returns:
            str: returns a new dataset corresponding to the converted input dataset on line 1, and a schema for its fields on line 2.

        You should show the user the fields of the new dataset, and the value of computed_from if specified. 
        """

        code = agent.context.get_code(
            "convert_dataset",
            {
                "input_dataset": input_dataset,
                "schema_name": schema_name,
                "cardinality": cardinality,
                "computed_from": computed_from if computed_from else None
            },
        )
        if PRINT_OUTPUT:
            print(code)

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )

            return self.handle_response(result)
        
    @tool
    async def retrieve_current_dataset_fields(
        self, agent: AgentRef, loop: LoopControllerRef
    ) -> str:
        """
        This function returns a schema for the current fields available on the active dataset.
        This function may be used when deciding how to specify a `computed_from` argument on a filter/convert operation to confirm the input fields for the operation.
        For example, if a user is converting the dataset using a TranslatePaper schema which includes 2 fields to translate the paper's abstract and body, you may want to check for the existence of these fields on the dataset.
        
        Returns:
            str: a dict representation of the schema for the current fields on the active dataset.
        """
        code = agent.context.get_code("retrieve_current_dataset_fields", {})
        if PRINT_OUTPUT:
            print(code)

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )
            return self.handle_response(result)
        
    @tool()
    async def backtrack_dataset_operation(
        self, agent: AgentRef, loop: LoopControllerRef
    ) -> str:
        """
        This function reverses the most recent dataset operation (convert_dataset, filter_dataset) back to the previous dataset.
        A dataset revision is only generated for successful operations, so this should only be used if the user requests you to undo an operation, not when an operation fails/errors.
        For example, if a user asks you to filter the dataset but then dislikes the filter you used, they may ask you to undo the filter, which can be done using this tool.
        
        Returns:
            str: returns the removed dataset operation and its arguments on line 1, the current dataset operation and its arguments on line 2, and a schema for its fields on line 3.
        """
        code = agent.context.get_code("backtrack_dataset_operation", {})
        if PRINT_OUTPUT:
            print(code)

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )
            return self.handle_response(result)

    @tool()
    async def set_input_dataset(
        self, dataset_name: str, agent: AgentRef, loop: LoopControllerRef
    ) -> str:
        """
        This function sets the input dataset for the agent to work with when using Palimpzest (pz).
        The dataset_name is the name of the dataset, for example the name of a folder, to set as the input source.
        Often, the dataset_name is defined after registering a dataset with the appropriate tool.
        The input source, also known as the source dataset, or the input dataset, is any dataset that the user will run any workload on.
        This function should be used at the beginning of any workflow to set the input dataset for the agent to work with when using Palimpzest (pz).

        Args:
            dataset_name (str): The name of the dataset that will be set as the input source.
        Returns:
            str: returns the input source dataset (including the detected file type) as a palimpzest dataset called `dataset` on line 1, and a schema for its fields on line 2.
        """

        code = agent.context.get_code(
            "set_input_dataset",
            {
                "dataset_name": dataset_name,
            },
        )
        if PRINT_OUTPUT:
            print(code)

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )
            return self.handle_response(result)

    @tool()
    async def list_schemas(self, agent: AgentRef) -> str:
        """
        This function lists all available schemas in the system. You should use these results to nicely format the output for the user.

        Returns:
            str: A table of the schemas in the system.
        """

        code = agent.context.get_code("list_schemas", {})
        if PRINT_OUTPUT:
            print(code)

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )

            return self.handle_response(result)

    @tool()
    async def execute_workload(
        self,
        output_dataset: str,
        policy_method: str,
        agent: AgentRef,
        loop: LoopControllerRef,
    ) -> str:
        """
        This function executes a workload starting from a given output dataset.
        If necessary, before executing the workload, any input dataset must be processed to match the schema of the output dataset.
        Processing an input dataset can be composed of several operations such as filtering or converting from one schema to the next. For example, if I want to extract the title of papers with at least 5 authors, I can first filter the papers to only include those with more than 5 authors and then convert the scientific papers to a schema that only includes the title information.
        In this case, the input dataset is the scientific papers dataset and the output dataset would be obtained first with filtering and then with converting the dataset to a schema that only includes the title information.

        The policy method chosen is either to minimize the extraction cost or to maximize the quality
        of the extraction.
        This returns the extractions as a Pandas DataFrame.

        Args:
            output_dataset (str): The dataset to execute the workload using.
            policy_method (str): Either "min_cost" or "max_quality". Defaults to "max_quality".

        Returns:
            str: returns the extracted references as a Pandas DataFrame called `results_df`.

        You should show the user the result after this function runs.

        """

        code = agent.context.get_code(
            "execute_workload",
            {
                "output_dataset": output_dataset,
                "policy_method": policy_method
            },
        )
        if PRINT_OUTPUT:
            print(code)

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )

            return self.handle_response(result)

    @tool()
    async def print_statistics(
        self,
        agent: AgentRef,
    ) -> str:
        """
        This function shows the runtime statistics after executing a workload.
        The function can be used to check the total cost and total runtime of the pipeline that was run.
        If necessary, before showing the statistics, the workload has to be executed.

        Returns:
            str: returns the statistics objects as it is produced by the execute workflow tool.

        You should show the user the result after this function runs.

        """

        code = agent.context.get_code(
            "print_statistics",
            {},
        )

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )
            return self.handle_response(result)
        
    @tool
    async def save_workload_results(
        self,
        output_path: str,
        agent: AgentRef
    ) -> str:
        """
        This function saves the results dataframe after executing a workload.
        This can be used if the user wants to save the results to their filesystem.
        The workload must be executed before using this tool to save its results.
        You should default to CSV format unless the user specifies otherwise.

        Args:
            output_path (str): The path to save the dataset to. Supported file extensions include csv, json, parquet, feather, XLSX/XLS, HTML, and tex.
        
        Returns:
            str: written file size in bytes.
        """

        code = agent.context.get_code(
            "save_workload_results",
            {
                "output_path": output_path
            },
        )

        if JSON_OUTPUT:
            return json.dumps(
                {
                    "action": "code_cell",
                    "language": "python3",
                    "content": code.strip(),
                }
            )
        else:
            result = await agent.context.evaluate(
                code,
                parent_header={},
            )
            return self.handle_response(result)

class BasicAgent(BaseAgent):
    """
    You are a helpful assistant designed to support users in working with Jupyter notebooks.
    Your role is to assist with analyzing data, automating tasks, organizing code, and helping users
    build and run notebook-based workflows more effectively.
    
    """

    async def auto_context(self):
        return """You are an intelligent assistant that helps users work more effectively in Jupyter notebooks.
Your role is to understand what the user is trying to accomplish—whether it's data analysis, visualization, coding, documentation, debugging, or exploration—and assist in completing that task efficiently within the notebook environment.

When responding:
- Break down complex requests into clear, executable steps.
- Suggest or generate code cells as needed.
- Offer explanations or alternatives where helpful.
- Anticipate follow-up steps and assist proactively.

Assume the user wants to make steady progress toward a goal (e.g. analyze a dataset, build a model, test a hypothesis, or explain a result). Provide helpful, minimal, and accurate code snippets. Maintain context between cells and help keep the workflow organized and reproducible.
"""