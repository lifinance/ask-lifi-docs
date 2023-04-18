import { Command, ux } from "@oclif/core";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { GitbookLoader } from 'langchain/document_loaders/web/gitbook'
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import * as dotenv from 'dotenv'
import { OpenAI } from "langchain/llms/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import * as chalk from 'chalk'
import * as fs from 'fs'
import { ChainValues } from "langchain/schema";

dotenv.config()

export default class AskCommand extends Command {
  static description = "Ask a question about LiFi";

  async run() {

    let vectorStore: HNSWLib

    if (!fs.existsSync('./.vectorstore/docstore.json')) {
      ux.log(chalk.blue('Loading docs from Gitbook...'))
      const loader = new GitbookLoader('https://docs.li.finance', { shouldLoadAllPaths: true })
      const docs = await loader.load()
      const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 0 })
      const splitDocs = await splitter.splitDocuments(docs)

      vectorStore = await HNSWLib.fromDocuments(splitDocs, new OpenAIEmbeddings())
      await vectorStore.save('./.vectorstore')
    } else {
      ux.action.start(chalk.blue('Loading'))
      vectorStore = await HNSWLib.load('./.vectorstore', new OpenAIEmbeddings())
      ux.action.stop()
    }

    const model = new OpenAI({})
    const chain = ConversationalRetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), { returnSourceDocuments: true })

    let chatHistory: string[] = []

    while (true) {
      const query = await ux.prompt(chalk.green('\n\nQuestion'))
      const res = await chain.call({ question: query, chat_history: chatHistory })

      ux.log(chalk.green('\nAnswer:\n'), res.text)
      ux.log(chalk.green('\nSources:'))
      chatHistory.push(query + '\n' + res.text)
      console.log(res.sourceDocuments.map((d: ChainValues) => d.metadata.source).join('\n'))
    }
  }
}
